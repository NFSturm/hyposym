module DrugInteractionStructs

    abstract type DrugCombination end

    struct PosComb <: DrugCombination end
    struct Other <: DrugCombination end

    struct Drug
        name::String
    end

    struct InteractionContext
        phrases::Vector{Vector{String}}
    end

    struct DrugInteraction
        drugs::Vector{Drug}
        context::InteractionContext
    end

    struct AnnotatedDrugInteraction
        interaction_type::DrugCombination
        interaction_specs::DrugInteraction
    end

    struct AnnotatedDrugCombination
        combination_type::DrugCombination
        drugs::Union{Vector{Drug}, Nothing}
    end

    struct DrugReport
        doc_id::String
        mentioned_drugs::Vector{Drug}
        drug_combinations::Union{Vector{Vector{AnnotatedDrugCombination}}, Vector{AnnotatedDrugCombination}}
        parsed_context
    end

end