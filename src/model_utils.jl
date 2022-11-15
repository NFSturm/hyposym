module ModelUtils

    using Dictionaries

    struct HypoSymLearner
        look_where::Vector{Int64}
        token_mixing_rate::Vector{Float64}
        recombination_prob::Float64
        recombination_percentage::Float64
    end

    struct PhraseDB
        phrases::Vector{Vector{String}}
        phrase_count::Dictionary{Vector{String}, Dictionary{Symbol, Int64}}
    end

    struct Model
        pos_comb_phrase_knowledge_base::PhraseDB
        other_phrase_knowledge_base::PhraseDB
        treatments
        hyperparams::HypoSymLearner
    end

    struct EvaluationReport
        recall::Float64
        precision::Float64
        f1::Float64
    end

    function flatten_drug_report_data(drug_report_data)
        drug_combinations =  []
    
        for drug_report in drug_report_data
            if !isnothing(drug_report.drug_combinations)
                for drug_combination in drug_report.drug_combinations
                    push!(drug_combinations, drug_combination)
                end
            end
        end
    
        drug_combinations
    end
    

    function evaluate_predictions(predictions, evaluator, combination_type)::EvaluationReport

        if isempty(predictions)
            return EvaluationReport(0.0000001, 0.0000001, 0.0000001)
        end

        true_pos_combs = [comb.drugs for comb in evaluator if comb.combination_type == combination_type]
        predicted_pos_combs = [comb.drugs for comb in predictions if comb.combination_type == combination_type]     

        num_matches = intersect!(Set(true_pos_combs), Set(predicted_pos_combs)) |> length

        # Compute evaluation metrics
        recall = round((num_matches / length(true_pos_combs)) + 0.0000001, digits=3)
        precision = round(num_matches / length(predicted_pos_combs) + 0.0000001, digits=3)
        f1 = round(2 * ((precision * recall) / ((precision + recall) + 0.0000001)), digits=3)

        EvaluationReport(recall, precision, f1)
    end


    function evaluate_predictions(model::Model, predictions, evaluator, combination_type)::EvaluationReport

        if isempty(predictions)
            return EvaluationReport(0.0000001, 0.0000001, 0.0000001)
        end

        true_pos_combs = [comb.drugs for comb in evaluator if comb.combination_type == combination_type]
        predicted_pos_combs = [comb.drugs for comb in predictions if comb.combination_type == combination_type]
        
        num_matches = intersect!(Set(true_pos_combs), Set(predicted_pos_combs)) |> length

        # Compute evaluation metrics
        recall = round((num_matches / length(true_pos_combs)) + 0.0000001, digits=3)
        precision = round(num_matches / length(predicted_pos_combs) + 0.0000001, digits=3)
        f1 = round(2 * ((precision * recall) / ((precision + recall) + 0.0000001)), digits=3)

        EvaluationReport(recall, precision, f1)
    end


    function evaluate_partial_predictions(predictions, evaluator, combination_type)::EvaluationReport

        if isempty(predictions)
            return EvaluationReport(0.0000001, 0.0000001, 0.0000001)
        end

        true_pos_combs = [comb.drugs for comb in evaluator if comb.combination_type == combination_type]
        predicted_pos_combs = [comb.drugs for comb in predictions if comb.combination_type == combination_type]     

        num_matches = []

        for true_drug_combination in true_pos_combs

            match_score = []

            for predicted_drug_combination in predicted_pos_combs
                if isequal(true_drug_combination, predicted_drug_combination)
                    push!(match_score, 1)
                elseif issubset(predicted_drug_combination, true_drug_combination)
                    partial_matches = [drug for drug in predicted_drug_combination if drug ∈ true_drug_combination]
                    partial_score = length(partial_matches) / length(true_drug_combination)
                    push!(match_score, partial_score)
                else
                    push!(match_score, 0)
                end
            end

            if !isempty(match_score)
                push!(num_matches, max(match_score...))
            end
        end

        # Compute evaluation metrics
        recall = round((sum(num_matches) / length(true_pos_combs)) + 0.0000001, digits=3)
        precision = round(sum(num_matches) / length(predicted_pos_combs) + 0.0000001, digits=3)
        f1 = round(2 * ((precision * recall) / ((precision + recall) + 0.0000001)), digits=3)

        EvaluationReport(recall, precision, f1)
    end


    function evaluate_partial_predictions(model::Model, predictions, evaluator, combination_type)::EvaluationReport

        if isempty(predictions)
            return EvaluationReport(0.0000001, 0.0000001, 0.0000001)
        end

        true_pos_combs = [comb.drugs for comb in evaluator if comb.combination_type == combination_type]
        predicted_pos_combs = [comb.drugs for comb in predictions if comb.combination_type == combination_type]     

        num_matches = []

        for true_drug_combination in true_pos_combs

            match_score = []

            for predicted_drug_combination in predicted_pos_combs
                if isequal(true_drug_combination, predicted_drug_combination)
                    push!(match_score, 1)
                elseif issubset(predicted_drug_combination, true_drug_combination)
                    partial_matches = [drug for drug in predicted_drug_combination if drug ∈ true_drug_combination]
                    partial_score = length(partial_matches) / length(true_drug_combination)
                    push!(match_score, partial_score)
                else
                    push!(match_score, 0)
                end
            end

            if !isempty(match_score)
                push!(num_matches, max(match_score...))
            end
        end

        # Compute evaluation metrics
        recall = round((sum(num_matches) / length(true_pos_combs)) + 0.0000001, digits=3)
        precision = round(sum(num_matches) / length(predicted_pos_combs) + 0.0000001, digits=3)
        f1 = round(2 * ((precision * recall) / ((precision + recall) + 0.0000001)), digits=3)

        EvaluationReport(recall, precision, f1)
    end

end