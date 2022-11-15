using Distributed
using Revise 

addprocs(5, topology=:master_worker, exeflags="--project=$(Base.active_project())")

@everywhere [include("stanza.jl"), include("jsonl.jl"), include("alexandria.jl"), include("context_generator.jl")]

@everywhere begin

    using Pkg; Pkg.instantiate()
    using Serialization: serialize

    using .Stanza
    using .JSONL
    using .Alexandria
    using .ContextGenerator: DrugReport, PosComb, Other, Drug, AnnotatedDrugCombination

    using MLStyle

end

drug_train = Alexandria.get_dataset("drug_interaction_train");

drug_train_data = JSONL.read_jsonl(drug_train["path"]);

mkdir("./context_paragraphs")

@everywhere begin
    
    function map_drug_interaction_type(drug_interaction)
        @match drug_interaction begin
            "POS" => PosComb()
            _ => Other()
        end
    end
    
    function parse_drug_report(drug_report)
        parsed_doc = Stanza.nlp(lowercase(drug_report["paragraph"]))
        doc_id = drug_report["doc_id"]
        mentioned_drugs = Drug.([lowercase.(drug["text"]) for drug in drug_report["spans"]])
    
        extractions = AnnotatedDrugCombination[]
    
        if !isempty(drug_report["rels"])
            relations = [(map_drug_interaction_type(relation["class"]), relation["spans"]) for relation in drug_report["rels"]]
            drug_id_mapping = Dict((id - 1 => lowercase.(drug["text"])) for (id, drug) in enumerate(drug_report["spans"]))
    
            for relation in relations
                comb_type, drug_ids = relation
                drug_ids_mapped = map(id -> drug_id_mapping[id], drug_ids)

                push!(extractions, AnnotatedDrugCombination(comb_type, Drug.(drug_ids_mapped)))
            end
        else
            push!(extractions, AnnotatedDrugCombination(Other(), nothing))
        end
    
        serialize("./context_paragraphs/$(doc_id).jls", DrugReport(doc_id, mentioned_drugs, extractions, parsed_doc))
    
    end
end

pmap(drug_report -> parse_drug_report(drug_report), drug_train_data)