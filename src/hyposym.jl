@doc """
This is the HypoSym main module.

This module serves as the execution environment for the HypoSym Learning System. 
"""

using Distributed
using Suppressor

@suppress begin
    addprocs(1, topology=:master_worker, exeflags="--project=$(Base.active_project())")

    @everywhere begin

        using Pkg; Pkg.instantiate()

        using Distributed
        using Distributions
        using Dictionaries
        using Statistics
        using StableRNGs
        using MLStyle
        using MLStyle.Modules.Cond
        using MLUtils
        using Chain
        using Pipe
        using IterTools
        using Parameters
        using StatsBase
        using Lazy
        using Combinatorics
        using Julog
        using Serialization
        using Logging
        using Revise
    end

    @everywhere [
        include("alexandria.jl"), 
        include("context_generator.jl"), 
        include("stanza.jl"),
        include("model_utils.jl")
    ]

    @everywhere begin

        using .Alexandria
        using .Stanza

        using .ModelUtils
        using .ModelUtils: HypoSymLearner,
            PhraseDB,
            Model,
            EvaluationReport,
            evaluate_predictions,
            evaluate_partial_predictions,
            flatten_drug_report_data

        using .ContextGenerator
        using .ContextGenerator: 
            Drug, 
            DrugInteraction, 
            AnnotatedDrugInteraction,
            PosComb, 
            Other,
            DrugCombination,
            InteractionContext,
            AnnotatedDrugCombination, 
            DrugReport
    end

end

function sample_mixing_rate(rng)
    upos_mix_rate = rand(rng, 0:0.1:1)
    lemma_mix_rate = 1 - upos_mix_rate
    [upos_mix_rate, lemma_mix_rate]
end

function get_random_params(rng)
    Dict(
        :look_where => [rand(rng, 1:6), rand(rng, 1:6)], 
        :token_mixing_rate => sample_mixing_rate(rng),
        :recombination_prob => round(rand(rng, Uniform(0,1), 1)[begin], digits=3),
        :recombination_percentage => round(rand(rng, Uniform(0, 0.4), 1)[begin], digits=3)
    )
end

function make_random_grid(rng, num_grid_elements::Int64)
    grid = Lazy.repeatedly(num_grid_elements, () -> get_random_params(rng))
    learner_params = [
        HypoSymLearner(
            pair[:look_where], 
            pair[:token_mixing_rate], 
            pair[:recombination_prob], 
            pair[:recombination_percentage]
        ) for pair in values(grid)]
    learner_params
end

map_drug_interaction_type(drug_interaction) = @match drug_interaction begin
    "POS" => PosComb()
    _ => Other()
end

@doc """
fold_datasets(rng::StableRNG, train_data, n_folds)

Folds ``train_data`` into ``n_folds``. 
"""
function fold_datasets(rng::StableRNG, train_data, n_folds)

    data_folds = []

    for fold_n in 1:n_folds
        seed_data, experience_data, population_validation_data = splitobs(shuffleobs(rng, train_data), at=(0.2, 0.5))
        push!(data_folds, (fold_n, seed_data, experience_data, population_validation_data))
    end

    data_folds
end

is_drug_conjunction_relation(drug_heads, mentioned_drugs) = all.(drug -> drug ∈ mentioned_drugs, drug_heads) 

instantiate_treatment(treatments)::Vector{Vector{Drug}} = [Drug.(treatment) for treatment in treatments]

function recombine_phrases(recombination_pairs, rng)

    recombined_pairs = []

    for pair in recombination_pairs
        head, tail = pair

        recombination_point = rand(rng, 1:lastindex(head))

        recombination_range_tail = @chain tail begin
            shuffleobs(rng, _)
            collect
            1:rand(rng, 2:lastindex(_))
        end

        recombined_pair = [head[begin:recombination_point], tail[recombination_range_tail]]

        push!(recombined_pairs, vcat(recombined_pair...))
    end
    recombined_pairs
end

function generate_phrase_recombinations(phrases, recombination_prob, recombination_percentage, rng)

    should_recombine = sample(rng, [true, false], Weights([recombination_prob, 1 - recombination_prob]))

    if should_recombine
        recombination_data, _ = splitobs(shuffleobs(rng, phrases), at=recombination_percentage)
        recombination_pairs = @chain recombination_data begin
            collect
            IterTools.partition(_, 2)
            collect
        end
        recombined_pairs = recombine_phrases(recombination_pairs, rng)
        return recombined_pairs
    else
        return nothing
    end
end

function load_drug_report_data(dir_name)
    @chain dir_name begin
        readdir(_, join=true)
        map(drug_report -> deserialize(drug_report), _)
    end
end

# --------------------------------- LEARNING BEGINS HERE ---------------------------------

# --------------------------------- SEED PHASE ---------------------------------

@doc """
seed_knowledge_base(seed_data, learner::HypoSymLearner, rng)

Iterates over the drug reports in ``seed_data`` and extract synergistic drug combinations as well
as the catch-call class "Other". Returns a ``PhraseDB`` for every class as well as a list of treatments
of type ``AnnotatedDrugCombination``.
"""
function seed_knowledge_base(seed_data, learner::HypoSymLearner, rng)

    # Go through datapoint in loop
    # Attribute PosComb Phrases and Other phrases
    # In the end, create knowledge base

    pos_contexts = []
    other_contexts = []
    treatments = AnnotatedDrugCombination[]

    for drug_report in seed_data
        # Extract drug combinations from seed data
        for drug_combination in drug_report.drug_combinations

            if !isnothing(drug_combination.drugs)

                # Generate context for learner from seeds                    
                interactions = ContextGenerator.generate_context_features(drug_report, drug_combination.drugs, learner, rng)
                
                for interaction in interactions
                    @match drug_combination.combination_type begin
                        ::PosComb => push!(pos_contexts, interaction.context.phrases)
                        ::Other => push!(other_contexts, interaction.context.phrases)
                    end
                end
                # Pushing treatments
                push!(treatments, drug_combination)
            else
                interactions = ContextGenerator.generate_context_features(drug_report, learner, rng)

                # Pushing "NoComb" context to "Other"
                for interaction in interactions
                    push!(other_contexts, interaction.context.phrases)
                end
            end
        end

    end

    # DB1: PosComb, DB2: Other
    # Generate DB: (phrases, num_occurences, max_epochs)

    other_contexts = Iterators.flatten(other_contexts) |> collect
    pos_contexts = Iterators.flatten(pos_contexts) |> collect

    other_phrase_counts = dictionary([phrase => dictionary([:num_occurences => 1, :num_epochs => 0]) for phrase in other_contexts]);
    pos_comb_phrase_counts = dictionary([phrase => dictionary([:num_occurences => 1, :num_epochs => 0]) for phrase in pos_contexts]);

    # # Updateable Knowledge Base for Learner
    pos_comb_phrase_db = PhraseDB(pos_contexts, pos_comb_phrase_counts);
    other_phrase_db = PhraseDB(other_contexts, other_phrase_counts);

    Model(pos_comb_phrase_db, other_phrase_db, treatments, learner)
end


# --------------------------------- END OF SEED PHASE ---------------------------------

# --------------------------------- NEW KNOWLEDGE INTRODUCTION ---------------------------------

function add_recombination_phrases(phrase_db::PhraseDB, learner::HypoSymLearner, rng)

    # Copy phrase db
    phrases = deepcopy(phrase_db.phrases)
    phrase_counts = deepcopy(phrase_db.phrase_count)

    # Recombining existing phrases
    recombined_phrases = generate_phrase_recombinations(phrases, learner.recombination_prob, learner.recombination_percentage, rng);

    if !isnothing(recombined_phrases)
        for recombined_phrase in recombined_phrases
            push!(phrases, recombined_phrase)
            set!(phrase_counts, recombined_phrase, dictionary([:num_occurrences => 0, :num_epochs => 0]))
        end

        return PhraseDB(phrases, phrase_counts)
    else
        return PhraseDB(phrases, phrase_counts)
    end

end

function add_new_phrases(phrase_db::PhraseDB, new_phrases::Vector{Vector{String}})

    # Copy phrase db
    phrases = deepcopy(phrase_db.phrases)
    phrase_counts = deepcopy(phrase_db.phrase_count)

    for new_phrase in new_phrases
        push!(phrases, new_prase)
        insert!(phrase_counts, new_phrase, dictionary([:num_occurrences => 0, :num_epochs => 0]))
    end

    PhraseDB(phrases, phrase_counts)
end


function update_phrase_database(phrase_db::PhraseDB, matches::Vector{Vector{String}})

    # Copy phrase db
    phrases = deepcopy(phrase_db.phrases)
    phrase_count = deepcopy(phrase_db.phrase_count)

    for match in matches
        num_occurrences, num_epochs = phrase_count[match]
        set!(phrase_count, match, dictionary([:num_occurrences => num_occurrences + 1, :num_epochs => num_epochs]))
    end

    PhraseDB(phrases, phrase_count)
end

function update_phrase_database(phrase_db::PhraseDB)

    # Copy phrase db
    phrases = deepcopy(phrase_db.phrases)
    phrase_count = deepcopy(phrase_db.phrase_count)

    for phrase in phrases
        num_occurrences, num_epochs = phrase_count[phrase]
        set!(phrase_count, phrase, dictionary([:num_occurrences => num_occurrences, :num_epochs => num_epochs + 1]))
    end    

    PhraseDB(phrases, phrase_count)
end


function add_new_treatment(existing_treatments::Vector{AnnotatedDrugCombination}, new_treatment::AnnotatedDrugCombination)
    all_treatments = deepcopy(existing_treatments)
    push!(all_treatments, new_treatment)

    all_treatments
end


function instantiate_initial_model(seed_data::Vector{DrugReport}, learner::HypoSymLearner, rng)

    model = seed_knowledge_base(seed_data, learner, rng)

    # Recombine phrases in "PosComb" and "Other" class
    model = Model(
        add_recombination_phrases(model.pos_comb_phrase_knowledge_base, learner, rng), 
        add_recombination_phrases(model.other_phrase_knowledge_base, learner, rng),
        model.treatments,
        learner
    )

    model

end


function instantiate_initial_models_for_fold(fold, param_grid, rng)::Vector{Model}

    initial_models = Model[]

    model_num = 1

    for learner_params in param_grid
        seed_data_structured = convert(Vector{DrugReport}, fold[2])
        instantiated_model = instantiate_initial_model(seed_data_structured, learner_params, rng)
        push!(initial_models, instantiated_model)

        @info "Model $(model_num) instantiated"
        model_num = model_num + 1
    end

    initial_models
end


# --------------------------------- GENERATE NEW PHRASES AND TREATMENTS FROM DRUG REPORT ---------------------------------

function generate_new_knowledge_from_drug_report(drug_report::DrugReport, learner::HypoSymLearner, rng)::Vector{DrugInteraction}
    # Generate context for new datapoint
    context_features_for_treatments::Vector{DrugInteraction} = ContextGenerator.generate_context_features(drug_report, learner, rng);

    # Return context features together with treatments
    context_features_for_treatments
end

# --------------------------------- EVALUATE KNOWLEDGE FROM UNSEEN DRUG REPORT ---------------------------------

function evaluate_new_knowledge_for_pos_comb(model::Model, new_knowledge::DrugInteraction)::Model

    pos_comb_phrase_knowledge_base = deepcopy(model.pos_comb_phrase_knowledge_base)
    treatments_knowledge_base = deepcopy(model.treatments)

    drug_combination_matches = intersect!(Set([treatment.drugs for treatment in treatments_knowledge_base]), Set(new_knowledge.drugs))
    phrase_matches = intersect!(Set(pos_comb_phrase_knowledge_base.phrases), Set(new_knowledge.context.phrases))

    # Update the corresponding model entry
    @match (isempty(drug_combination_matches), isempty(phrase_matches)) begin
        (true, true) => begin
            return Model(
                pos_comb_phrase_knowledge_base, 
                model.other_phrase_knowledge_base, 
                treatments_knowledge_base, 
                model.hyperparams
            )
        end
        (true, false) => begin
            treatments_knowledge_base = add_new_treatment(treatments_knowledge_base, AnnotatedDrugCombination(PosComb(), new_knowledge.drugs)) 
            pos_comb_phrase_knowledge_base = update_phrase_database(pos_comb_phrase_knowledge_base, phrase_matches |> collect)
            
            return Model(
                pos_comb_phrase_knowledge_base,
                model.other_phrase_knowledge_base, 
                treatments_knowledge_base, 
                model.hyperparams
            )
        end
        (false, true) => begin
            pos_comb_phrase_knowledge_base = add_new_phrases(pos_comb_phrase_knowledge_base, new_knowledge.context.phrases)

            return Model(
                pos_comb_phrase_knowledge_base,
                model.other_phrase_knowledge_base, 
                treatments_knowledge_base, 
                model.hyperparams
            )
        end
        (false, false) => begin
            pos_comb_phrase_knowledge_base = update_phrase_database(pos_comb_phrase_knowledge_base, phrase_matches |> collect)

            return Model(
                pos_comb_phrase_knowledge_base,
                model.other_phrase_knowledge_base, 
                treatments_knowledge_base, 
                model.hyperparams
            )
        end
    end
end


function evaluate_new_knowledge_for_other(model::Model, new_knowledge::DrugInteraction)::Model

    other_phrase_knowledge_base = deepcopy(model.other_phrase_knowledge_base)
    treatments_knowledge_base = deepcopy(model.treatments)

    drug_combination_matches = intersect!(Set([treatment.drugs for treatment in treatments_knowledge_base]), Set(new_knowledge.drugs))
    phrase_matches = intersect!(Set(other_phrase_knowledge_base.phrases), Set(new_knowledge.context.phrases))

    # Update the corresponding model entry
    @match (isempty(drug_combination_matches), isempty(phrase_matches)) begin
        (true, true) => begin

            return Model(
                model.pos_comb_phrase_knowledge_base, 
                other_phrase_knowledge_base, 
                treatments_knowledge_base, 
                model.hyperparams
            )
        end
        (true, false) => begin
            treatments_knowledge_base = add_new_treatment(treatments_knowledge_base, AnnotatedDrugCombination(Other(), new_knowledge.drugs)) 
            other_phrase_knowledge_base = update_phrase_database(other_phrase_knowledge_base, phrase_matches |> collect)

            return Model(
                model.pos_comb_phrase_knowledge_base, 
                other_phrase_knowledge_base, 
                treatments_knowledge_base, 
                model.hyperparams
            )
        end
        (false, true) => begin
            other_phrase_knowledge_base = add_new_phrases(other_phrase_knowledge_base, new_knowledge.context.phrases)

            return Model(
                model.pos_comb_phrase_knowledge_base, 
                other_phrase_knowledge_base, 
                treatments_knowledge_base, 
                model.hyperparams
            )
        end
        (false, false) => begin
            other_phrase_knowledge_base = update_phrase_database(other_phrase_knowledge_base, phrase_matches |> collect)

            return Model(
                model.pos_comb_phrase_knowledge_base, 
                other_phrase_knowledge_base, 
                treatments_knowledge_base, 
                model.hyperparams
            )
        end
    end
end


function predict_drug_combination(model::Model, new_datapoint::DrugInteraction)

    other_phrase_knowledge_base = deepcopy(model.other_phrase_knowledge_base)
    pos_comb_phrase_knowledge_base = deepcopy(model.pos_comb_phrase_knowledge_base)
    treatments_knowledge_base = deepcopy(model.treatments)

    pos_comb_treatments = [treatment for treatment in treatments_knowledge_base if treatment.combination_type == PosComb()]
    other = [treatment for treatment in treatments_knowledge_base if treatment.combination_type == Other()]

    for treatment_type in [(PosComb(), pos_comb_treatments, pos_comb_phrase_knowledge_base), (Other(), other, other_phrase_knowledge_base)]

        interaction_type, treatments, phrase_knowledge_base = treatment_type

        drug_combination_matches = intersect!(Set(treatments), Set(new_datapoint.drugs))
        phrase_matches = intersect!(Set(phrase_knowledge_base.phrases), Set(new_datapoint.context.phrases)) 
        
        @match (isempty(drug_combination_matches), isempty(phrase_matches)) begin
            (true, true) => begin
                return Model(
                    model.pos_comb_phrase_knowledge_base, 
                    other_phrase_knowledge_base, 
                    treatments_knowledge_base, 
                    model.hyperparams
                )
            end
            (_, _) => begin
                return model, AnnotatedDrugCombination(interaction_type, new_datapoint.drugs)
            end
        end
    end
end


function predict_drug_combinations(model::Model, new_datapoints::Vector{DrugReport}, rng)

    predictions = []

    for new_datapoint in new_datapoints

        extracted_drug_interactions = ContextGenerator.generate_context_features(new_datapoint, model.hyperparams, rng)

        for drug_interaction in extracted_drug_interactions

            prediction = predict_drug_combination(model, drug_interaction)

            @match prediction begin
                (m :: Model, dc :: AnnotatedDrugCombination) => begin
                    push!(predictions, dc)
                end
                new_model :: Model => begin
                    model = new_model
                end
            end

        end
    end
    predictions
end

# --------------------------------- END OF NEW KNOWLEDGE INTRODUCTION ---------------------------------

function combine_models(model1::Model, model2::Model)::Model
    Model(
        PhraseDB(
            vcat(model1.pos_comb_phrase_knowledge_base.phrases, model2.pos_comb_phrase_knowledge_base.phrases),
            merge!(model1.pos_comb_phrase_knowledge_base.phrase_count, model2.pos_comb_phrase_knowledge_base.phrase_count)
        ),
        PhraseDB(
            vcat(model1.other_phrase_knowledge_base.phrases, model2.other_phrase_knowledge_base.phrases),
            merge!(model1.other_phrase_knowledge_base.phrase_count, model2.other_phrase_knowledge_base.phrase_count)
        ),
        vcat(model1.treatments, model2.treatments),
        model1.hyperparams # Take the hyperparams of the better model
    )
end

# --------------------------------- TRAINING LOOP ---------------------------------

@doc """
train_models(models::Vector{Models}, fold, max_epochs, rng)

``max_epochs`` needs to be divisible by 5. Model training stops after at most ``max_epochs``
because of data partitioning. 
"""
function train_models(models::Vector{Model}, fold, max_epochs, rng, exact_match = true)

    fold_num, experience_data, population_validation_data = fold

    # Converting data into proper type
    experience_data = convert(Vector{DrugReport}, experience_data)
    population_validation_data = convert(Vector{DrugReport}, population_validation_data)

    # Partitioning experience data by epoch
    experience_partition_by_epoch = @chain experience_data begin
        chunk(_, max_epochs)
        convert.(Vector{DrugReport}, _)
    end

    validation_data_by_epoch = @chain population_validation_data begin
        chunk(_, Int(floor(max_epochs / 5)))
        convert.(Vector{DrugReport}, _)
    end

    # Dictionaries to tract best model performance
    performance_history = []

    @info "Training models for $(lastindex(experience_partition_by_epoch)) epochs."

    for epoch in 1:lastindex(experience_partition_by_epoch)

        if length(models) < 16
            break
        else
            updated_models = Model[]

            if !(epoch % 5 == 0)
                for model in models

                    new_experiences = experience_partition_by_epoch[epoch]

                    generated_knowledge = Iterators.flatten([generate_new_knowledge_from_drug_report(new_experience, model.hyperparams, rng) for new_experience in new_experiences]) |> collect

                    # Updating the model for PosComb
                    for knowledge in generated_knowledge
                        model = evaluate_new_knowledge_for_pos_comb(model, knowledge)
                    end

                    # Updating the model for Other
                    for knowledge in generated_knowledge
                        model = evaluate_new_knowledge_for_other(model, knowledge)
                    end

                    model = Model(
                        add_recombination_phrases(model.pos_comb_phrase_knowledge_base, model.hyperparams, rng), 
                        add_recombination_phrases(model.other_phrase_knowledge_base, model.hyperparams, rng),
                        model.treatments,
                        model.hyperparams
                    )

                    push!(updated_models, model)

                    models = updated_models
                end

            elseif epoch % 5 == 0

                if Int(floor(epoch / 5)) > lastindex(validation_data_by_epoch)
                    break
                else
                    for model in models

                        new_experiences = experience_partition_by_epoch[epoch]
    
                        generated_knowledge = Iterators.flatten([generate_new_knowledge_from_drug_report(new_experience, model.hyperparams, rng) for new_experience in new_experiences]) |> collect
    
                        for knowledge in generated_knowledge
                            model = evaluate_new_knowledge_for_pos_comb(model, knowledge)
                        end
    
                        for knowledge in generated_knowledge
                            model = evaluate_new_knowledge_for_other(model, knowledge)
                        end
    
                        model = Model(
                            add_recombination_phrases(model.pos_comb_phrase_knowledge_base, model.hyperparams, rng), 
                            add_recombination_phrases(model.other_phrase_knowledge_base, model.hyperparams, rng),
                            model.treatments,
                            model.hyperparams
                        )
    
                        push!(updated_models, model) 
                    end
    
                    models = updated_models
    
                    # Selecting appropriate validation_data
                    validation_data = validation_data_by_epoch[Int(floor(epoch / 5))]
                    validation_data_flattened = flatten_drug_report_data(validation_data)
    
                    validation_data_for_prediction = convert(Vector{DrugReport}, validation_data)
    
                    # Making predictions
                    predictions_per_model = [predict_drug_combinations(model, validation_data_for_prediction, rng) for model in models]
    
                    # Evaluate prediction with model index
                    evaluated_predictions_exact_match = [(index, evaluate_predictions(predictions, validation_data_flattened, PosComb())) for (index, predictions) in enumerate(predictions_per_model)]
    
                    evaluated_predictions_partial_match = [(index, evaluate_partial_predictions(predictions, validation_data_flattened, PosComb())) for (index, predictions) in enumerate(predictions_per_model)]
    
                    # Order prediction results
                    # Select best + randomly sampled
    
                    # Selection strategy: Discard worst 25% of models. From those 25%, select 10% at random to keep in the population_validation_data
                    # Since all learners have access to the same data and "cross-over" takes place inside a learner, the learners work independently until validation epochs
                    
                    # Important! Reverse sorting, otherwise worst models are put at the front 
                    sort!(evaluated_predictions_exact_match, by = evaluation -> evaluation[end].f1, rev = true)
                    sort!(evaluated_predictions_partial_match, by = evaluation -> evaluation[end].f1, rev = true)
    
                    @match exact_match begin
                        true => @info "Best model in $(epoch): $(evaluated_predictions_exact_match[begin][end].f1)"
                        false => @info "Best model in $(epoch): $(evaluated_predictions_partial_match[begin][end].f1)"
                    end

                    if exact_match
                        # Use lastindex() because length() function is not implemented for type Vector{EvaluationReport}
                        best_quarters_exact_match_report_indices, worst_quarter_exact_match_report_indices = splitobs(evaluated_predictions_exact_match, at=0.75) |> collect
                        
                        # Retrieving model indices
                        best_quarters_exact_match = [exact_match_report[begin] for exact_match_report in best_quarters_exact_match_report_indices]
                        worst_quarter_exact_match  = [exact_match_report[begin] for exact_match_report in worst_quarter_exact_match_report_indices]

                        random_model_idx, _ = splitobs(shuffleobs(rng, worst_quarter_exact_match), at=0.1) |> collect
    
                        best_models = models[best_quarters_exact_match]
                        random_models = models[random_model_idx]

                        # Elitist merge of 10 best models
                        model_chunks = chunk(best_models[1:10], size=2)
    
                        best_models_combined = [combine_models(model_chunk[1], model_chunk[2]) for model_chunk in model_chunks]
    
                        new_generation_models = Iterators.flatten([best_models_combined, best_models[11:end], random_models]) |> collect
    
                        models = new_generation_models
    
                        push!(performance_history, Dict(epoch => [evaluated_predictions_exact_match[1:2]]))
    
                    else
                        best_quarters_partial_match_report_indices, worst_quarter_partial_match_report_indices = splitobs(evaluated_predictions_partial_match, at=0.75) |> collect
                        
                        # Retrieving model indices
                        best_quarters_partial_match = [partial_match_report[begin] for partial_match_report in best_quarters_partial_match_report_indices]
                        worst_quarter_partial_match  = [partial_match_report[begin] for partial_match_report in worst_quarter_partial_match_report_indices]

                        random_model_idx, _ = splitobs(shuffleobs(rng, worst_quarter_partial_match), at=0.1) |> collect
    
                        best_models = models[best_quarters_partial_match]
                        random_models = models[random_model_idx]
    
                        # Elitist merge of 10 best models
                        model_chunks = chunk(best_models[1:10], size=2)
    
                        best_models_combined = [combine_models(model_chunk[1], model_chunk[2]) for model_chunk in model_chunks]
        
                        new_generation_models = Iterators.flatten([best_models_combined, best_models[11:end], random_models]) |> collect
    
                        models = new_generation_models
    
                        push!(performance_history, Dict(epoch => [evaluated_predictions_partial_match[1:2]]))
                    end    
                end
            end
        end
        @info "Epoch $epoch concluded. $(length(models)) models remaining."
    end

    # Return best 10 models
    fold_num, models[1:10], performance_history
end

# --------------------------------- END OF TRAINING LOOP ---------------------------------

# --------------------------------- EXECUTION ---------------------------------

#const rng = StableRNG(28101997)
rng = StableRNG(28101997);

drug_report_data = load_drug_report_data("./context_paragraphs");

# 85% training - 15% validation
train_data, validation_data = splitobs(shuffleobs(rng, drug_report_data), at=0.85) |> collect;

folds = fold_datasets(rng, train_data, 5);

grid = make_random_grid(rng, 150);

# Using five separate folds for training
for fold in folds

    @info "Starting fold number $(fold[1])."

    @info "Instantiating base models"
    models = instantiate_initial_models_for_fold(fold, grid, rng);

    @info "Training Models for Exact Matches"
    results = train_models(models, fold, 30, rng);

    @info "Serializing model results in fold $(fold[1])"
    serialize("./model_results_exact_fold_$(fold[1]).jls", results);

    @info "Training Models for Partial Matches"
    partial_results = train_models(models, fold, 30, rng, false);

    @info "Serializing model results for partial model in fold $(fold[1])"
    serialize("./model_results_partial_fold_$(fold[1]).jls", partial_results);

    @info "Finished fold number $(fold[1])."
end

# --------------------------------- EVALUATION ON VALIDATION AND TEST SET ---------------------------------

# Loading test data
drug_report_data_test = load_drug_report_data("./context_paragraphs_test");
test_data = flatten_drug_report_data(drug_report_data_test);

#validation_data = deserialize("./validation_data.jls");
serialize("./validation_data.jls", validation_data);
validation_data_structured = convert(Vector{DrugReport}, validation_data);

best_exact_match_model_per_fold = []
best_exact_match_hyperparams = []
exact_match_test_set = []

best_partial_match_model_per_fold = []
best_partial_match_hyperparams = []
partial_match_test_set = []    

for fold_num in 1:5

    @info "Deserializing fold results in fold $(fold_num)"
    # Exact Match
    exact_fold_results = deserialize("./model_results_exact_fold_$(fold_num).jls")

    exact_match_models = [model for model in exact_fold_results[2]]

    predictions_per_model = []

    i = 1
    for model in exact_match_models
        predictions = predict_drug_combinations(model, validation_data_structured, rng)
        push!(predictions_per_model, predictions)

        @info "Predictions concluded for exact match model $(i) in fold $(fold_num)."
        i = i + 1
    end

    evaluated_predictions_exact_match = [(index, evaluate_predictions(predictions, flatten_drug_report_data(validation_data), PosComb())) for (index, predictions) in enumerate(predictions_per_model)]
    sort!(evaluated_predictions_exact_match, by = evaluation -> evaluation[end].f1, rev = true)

    best_model_exact_match_idx = evaluated_predictions_exact_match |> first |> first
    best_model_performance_validation_set_exact = evaluated_predictions_exact_match |> first |> last

    @show "Best exact match model fold $(fold_num):"

    @show exact_match_models[best_model_exact_match_idx].hyperparams
    push!(best_exact_match_hyperparams, exact_match_models[best_model_exact_match_idx].hyperparams)

    @show "Best performance for exact match model on validation set in fold $(fold_num):" 
    @show best_model_performance_validation_set_exact

    push!(best_exact_match_model_per_fold, best_model_performance_validation_set_exact)

    test_set_predictions = predict_drug_combinations(exact_match_models[best_model_exact_match_idx], drug_report_data_test, rng)
    test_set_evaluation = evaluate_predictions(test_set_predictions, flatten_drug_report_data(drug_report_data_test), PosComb())

    @show "Prediction Score (F1) for Exact Matches on Test Set in fold $(fold_num):"
    @show test_set_evaluation
    push!(exact_match_test_set, test_set_evaluation)

    # Partial Match
    partial_fold_results = deserialize("./model_results_partial_fold_$(fold_num).jls")

    partial_match_models = [model for model in partial_fold_results[2]]

    predictions_per_model = []
    
    i = 1
    for model in partial_match_models
        predictions = predict_drug_combinations(model, validation_data_structured, rng)
        push!(predictions_per_model, predictions)

        @info "Predictions concluded for partial match model $(i) in fold $(fold_num)."
        i = i + 1
    end

    evaluated_predictions_partial_match = [(index, evaluate_predictions(predictions, flatten_drug_report_data(validation_data), PosComb())) for (index, predictions) in enumerate(predictions_per_model)]
    sort!(evaluated_predictions_partial_match, by = evaluation -> evaluation[end].f1, rev = true)

    best_model_partial_match_idx = evaluated_predictions_partial_match |> first |> first
    best_model_performance_validation_set_partial = evaluated_predictions_partial_match |> first |> last

    @show "Best partial match model fold $(fold_num):" 
    @show partial_match_models[best_model_partial_match_idx].hyperparams
    push!(best_partial_match_hyperparams, partial_match_models[best_model_partial_match_idx].hyperparams)

    @show "Best performance for partial match model on validation set in fold $(fold_num):"
    @show best_model_performance_validation_set_partial

    push!(best_partial_match_model_per_fold, best_model_performance_validation_set_partial)

    test_set_predictions = predict_drug_combinations(partial_match_models[best_model_partial_match_idx], drug_report_data_test, rng)
    test_set_evaluation = evaluate_predictions(test_set_predictions, flatten_drug_report_data(drug_report_data_test), PosComb())

    @show "Prediction Score (F1) for Partial Matches on Test Set in fold $(fold_num):"
    @show test_set_evaluation
    push!(partial_match_test_set, test_set_evaluation)

end