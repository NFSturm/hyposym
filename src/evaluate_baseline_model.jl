include("alexandria.jl")
include("context_generator.jl")
include("stanza.jl")
include("model_utils.jl")

using .ModelUtils
using .ModelUtils: EvaluationReport, 
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
    
using MLUtils
using StableRNGs
using MLStyle
using Chain
using Serialization
using Revise


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

function load_drug_report_data(dir_name)
    @chain dir_name begin
        readdir(_, join=true)
        map(drug_report -> deserialize(drug_report), _)
    end
end

function predict_with_rule_based(document, pattern::Vector{String}, drugs)
    
    drug_combinations = []

    drugs_stringified = [drug.name for drug in drugs]

    for sentence in document.parsed_context.sentences
    
        mentioned_drugs_in_sentence = [drug for drug in drugs_stringified if occursin(drug, lowercase(sentence.text))]

        if (length(mentioned_drugs_in_sentence) >= 2) && any(occursin.(pattern, lowercase(sentence.text)))
            # Extract drugs in sentence
            drugs_in_sentence = [drug for drug in drugs if occursin(drug.name, lowercase(sentence.text))] |> unique
            push!(drug_combinations, AnnotatedDrugCombination(PosComb(), drugs_in_sentence))
        else
            continue
        end
    end

    drug_combinations
end


function predict_baseline_model(drug_reports::Vector{DrugReport}, trigger_patterns::Vector{String})

    drug_combinations = []

    for drug_report in drug_reports
        triggered_drug_combinations = predict_with_rule_based(drug_report, trigger_patterns, drug_report.mentioned_drugs)    
        push!(drug_combinations, triggered_drug_combinations)
    end

    Iterators.flatten(drug_combinations) |> collect
end

# --------------------------------- Instantiating constants ---------------------------------

const rng = StableRNG(28101997)

const trigger_patterns = [
    "combination",
    "plus",
    "combined",
    "followed by",
    "first-line",
    "combinations",
    "prior to",
    "synergistic",
    "beneficial",
    "combining",
    "sequential",
    "additive",
    "synergy",
    "first line",
    "synergism",
    "conjunction",
    "two-drug",
    "first choice",
    "additivity",
    "combinational",
    "synergetic",
    "simultaneously with",
    "supra-additive",
    "five-drug",
    "combinatory",
    "over-additive",
    "timed-sequential",
    "co-blister",
    "super-additive",
    "synergisms",
    "synergic",
    "synergistical",
    "less-than-additive",
    "greater-than-additive",
    "additivesynergistic",
    "supraadditive",
    "superadditive",
    "overadditive",
    "subadditive",
    "first-choice",
    "2-drug",
    "sub-additive",
    "more-than-additive",
    "3-drug"
]

# --------------------------------- EVALUATION OF BASELINE MODEL ---------------------------------

# --------------------------------- EVALUATION OF FOLD VALIDATION DATA ---------------------------------
drug_report_data = load_drug_report_data("./context_paragraphs");

folds = fold_datasets(rng, drug_report_data, 5);

validation_data = [convert(Vector{DrugReport}, fold[4]) for fold in folds];

predictions_for_folds = [predict_baseline_model(convert(Vector{DrugReport}, validation_data_fold), trigger_patterns) for validation_data_fold in validation_data];

evaluation_reports_validation_data = [evaluate_predictions(fold_predictions, flatten_drug_report_data(validation_data_fold), PosComb()) for (fold_predictions, validation_data_fold) in zip(predictions_for_folds, validation_data)];
evaluation_reports_validation_data_partial_matches = [evaluate_partial_predictions(fold_predictions, flatten_drug_report_data(validation_data_fold), PosComb()) for (fold_predictions, validation_data_fold) in zip(predictions_for_folds, validation_data)];

# --------------------------------- EVALUATION OF TEST DATA ---------------------------------
drug_report_data_test = load_drug_report_data("./context_paragraphs_test");

test_data = flatten_drug_report_data(drug_report_data_test);

predictions_test_set = predict_baseline_model(drug_report_data_test, trigger_patterns);

test_results = evaluate_predictions(predictions_test_set, test_data, PosComb());
test_results_partial_matches = evaluate_partial_predictions(predictions_test_set, test_data, PosComb());