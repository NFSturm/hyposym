module ContextGenerator
    include("stanza.jl")
    include("drug_interaction_structs.jl")

    using .Stanza
    using .DrugInteractionStructs: Drug, 
        DrugInteraction, 
        AnnotatedDrugInteraction,
        PosComb, 
        Other,
        DrugCombination,
        InteractionContext,
        AnnotatedDrugCombination,
        DrugReport

    using MLStyle
    using MLStyle.Modules.Cond
    using StatsBase
    using Lazy
    using Combinatorics
    using Chain
    using Pipe
    using IterTools
    using Statistics
    using MLUtils

    get_drugs_for_datapoint(datapoint) = [lowercase.(drug["text"]) for drug in datapoint["spans"]]

    parse_word_head(head::Int64, id::Int64) = head == 0 ? id : head

    function leave_unchanged_lemma(word)
        if isnothing(word.lemma)
            lowercase(word.text)
        else
            length(word.text) == 1 ? lowercase(word.text) : lowercase(word.lemma)
        end
    end
        

    function map_drug_synonyms_from_expression(drug, synonym_candidate, sentence)

        candidate_sentence = replace(lowercase(sentence), r"\s{2,}" => " ")
    
        @cond begin
            occursin("$(drug) ($(synonym_candidate))", candidate_sentence) => drug => synonym_candidate
            occursin("$(drug) ($(synonym_candidate))", candidate_sentence) => drug => synonym_candidate
            occursin("$(synonym_candidate) ( $(drug) )", candidate_sentence) => synonym_candidate => drug
            occursin("$(synonym_candidate) ($(drug))", candidate_sentence) => synonym_candidate => drug
            _ => drug => drug
        end
    end
    
    
    function traverse_drug_pointers(drug_id, drug_name, pointer::Int64, drug_pointers_lookup, mentioned_drugs, drug_synonym_mapping)
    
        # Initialize return variable
        final_lemma = nothing
        final_index = nothing
    
        while true
            maybe_drug = drug_pointers_lookup[pointer]
    
            if !isempty(drug_synonym_mapping)
                maybe_drug["lemma"] = replace(maybe_drug["lemma"], maybe_drug["lemma"] => get(drug_synonym_mapping, maybe_drug["lemma"], maybe_drug["lemma"]))
            end
    
            if maybe_drug["head"] == 0
                final_lemma = drug_name
                final_index = drug_id
                break
            elseif maybe_drug["lemma"] ∈ getfield.(mentioned_drugs, :name)
                final_lemma = maybe_drug["lemma"]
                final_index = maybe_drug["id"]
                break
            else
                pointer = maybe_drug["head"]
            end
        end
    
        drug_id, drug_name, final_index, final_lemma
    end
    
    
    function identify_treatments(drug_dependencies::Vector{Tuple{Int64, String, Int64, String}}, first_drug_mention::Int64)
    
        treatments = []
        drug_pointer = 1
        chaining_found = nothing
    
        treatment = []
    
        while drug_pointer <= length(drug_dependencies)
    
            if !isnothing(chaining_found)
                treatment = copy(last(treatments)) # Copy, otherwise ``treatments`` is mutated
                chaining_found = nothing
            end
    
            drug_id, drug_name, dependency_id, dependency_name = drug_dependencies[drug_pointer]
    
            if (drug_id == dependency_id) && (dependency_id == first_drug_mention)
                push!(treatment, (drug_id, drug_name))
            elseif (drug_name == dependency_name) && (drug_id == dependency_id) && ((drug_id, drug_name) ∉ [drug_tuple for drug_tuple in treatment])
                push!(treatments, treatment)
                push!(treatments, [(drug_id, drug_name)])
                chaining_found = true
            elseif (drug_name == dependency_name) && (drug_id != dependency_id) && ((drug_id, drug_name) ∉ [drug_tuple for drug_tuple in treatment])
                push!(treatments, treatment)
                chaining_found = true
            elseif (drug_name == dependency_name) && (drug_id != dependency_id) && ((drug_id, drug_name) ∉ [drug_tuple for drug_tuple in treatment])
                push!(treatments, treatment)
                chaining_found = true
            elseif (drug_name != dependency_name) && (drug_id != dependency_id) && (drug_name ∈ [drug_tuple[2] for drug_tuple in treatment]) && (drug_id ∉ [drug_tuple[1] for drug_tuple in treatment])
                push!(treatments, treatment)
                chaining_found = true
            elseif (drug_name != dependency_name) && (drug_id != dependency_id)
                push!(treatment, (drug_id, drug_name))
            end
    
            # Appends the treatment once the last index is reached
            if drug_pointer == length(drug_dependencies)
                push!(treatments, treatment)
            end
    
            drug_pointer = +(1, drug_pointer)
        end
    
        identified_treatments = @chain treatments begin
            filter(treatment -> lastindex(treatment) > 1, _) # Removes singletons
            map(t -> getindex.(t, 2), _)
            collect
            unique
        end
    
        @cond begin
            isempty(identified_treatments) => nothing
            _ => identified_treatments
        end
    
    end
    
    
    function identify_treatments_in_sentence(sentence, mentioned_drugs)
    
        drug_pointers_lookup = Dict(word.id => Dict("lemma" => leave_unchanged_lemma(word), "head" => word.head, "id" => word.id) for word in sentence.words)
    
        drug_with_pointers = [(word.id, leave_unchanged_lemma(word), parse_word_head(word.head, word.id)) for word in sentence.words if word.lemma ∈ getfield.(mentioned_drugs, :name)]
    
        sent_words = [lowercase(word.text) for word in sentence.words]
        
        drug_heads = [(word.lemma, sent_words[parse_word_head(word.head, word.id)]) for word in sentence.words if (any(occursin.(word.text, getfield.(mentioned_drugs, :name))) || any(occursin.(getfield.(mentioned_drugs, :name), word.text))) && length(word.text) > 2]
    
        drug_synonym_mapping = filter(!isnothing, [map_drug_synonyms_from_expression(drug_head[1], drug_head[2], sentence.text) for drug_head in drug_heads]) |> collect |> Dict
    
        drug_dependencies = [traverse_drug_pointers(drug_id, drug_name, pointer, drug_pointers_lookup, mentioned_drugs, drug_synonym_mapping) for (drug_id, drug_name, pointer) in drug_with_pointers]
        
        if !isempty(drug_dependencies)
            first_drug_mention = sort(drug_dependencies, by = drug_tuple -> drug_tuple[begin]) |> first |> first
            identify_treatments(drug_dependencies, first_drug_mention)  
        else
            nothing
        end
        
    end
    
    
    function identify_all_treatments_in_report(doc, mentioned_drugs)
        @chain doc.sentences begin
            [identify_treatments_in_sentence(sentence, mentioned_drugs) for sentence in _]
            filter(!isnothing, _)
            Iterators.flatten
            collect
            unique
        end
    end
    
    function find_related_pattern_constituents(pattern)
        @match pattern begin
            ((name1, upos1, id1, head1), (name2, upos2, id2, head2)) => head1 == id2 ? ((name1, upos1), (name2, upos2)) : nothing
            ((name, upos, id, head), ) => ((name, upos))
        end
    end
    
    function generate_variable_pattern_string(sentence, mentioned_drugs, width::Int64)
        candidates = @chain sentence.words begin
            [(leave_unchanged_lemma(word), word.upos, word.id, parse_word_head(word.head, word.id)) for word in _ if (word.upos ∈ ["NOUN", "PROPN", "PRON"]) && (word.lemma ∉ getfield.(mentioned_drugs, :name))]
            IterTools.partition(_, width, 1)
            collect
        end
    
        maybe_candidates = []
    
        for candidate in candidates
            maybe_candidate = find_related_pattern_constituents(candidate)
            push!(maybe_candidates, maybe_candidate)
        end
    
        filter(!isnothing, maybe_candidates)
    end
    
    function construct_document_graph_for_string_pattern(document, pattern::Vector{String})
    
        document_graph_init = []
    
            for sentence in document.sentences

                if all(occursin.(pattern, Ref(lowercase(sentence.text))))
                    push!(document_graph_init, 1)
                elseif (sum(occursin.(pattern, Ref(lowercase(sentence.text)))) >= length(pattern) - 1) && length(pattern) <= 3
                    push!(document_graph_init, 1)
                elseif (sum(occursin.(pattern, Ref(lowercase(sentence.text)))) >= length(pattern) - 2) && length(pattern) > 3
                    push!(document_graph_init, 1)
                else
                    push!(document_graph_init, 0)
                end
            end
        document_graph_init
    end
    
    function construct_document_graph_for_string_pattern(document, pattern)
    
        document_graph_init = []
    
            for sentence in document.sentences
                @cond begin
                    occursin(pattern, lowercase(sentence.text)) => push!(document_graph_init, 1)
                    _ => push!(document_graph_init, 0)
                end
            end
        document_graph_init
    end
    
    
    function construct_document_graph_for_string_patterns(document, patterns)
    
        document_graphs_for_variable_pattern = []
    
        for variable_pattern in patterns
            pattern_words = @match variable_pattern begin
                ((name1, upos1), (name2, upos2)) => join([name1, name2], " ")
                (name, upos) => name
            end
            document_graph_for_variable_pattern = construct_document_graph_for_string_pattern(document, pattern_words)
    
            if !all(elem -> elem == 0, document_graph_for_variable_pattern)
                push!(document_graphs_for_variable_pattern, (document_graph_for_variable_pattern, variable_pattern))
            end
        end
    
        filter(pattern_occurrence -> sum(first(pattern_occurrence)) > 1, document_graphs_for_variable_pattern) |> collect
    
    end
    
    
    generate_variable_pattern_strings_for_sentence(sentence, mentioned_drugs, max_width::Int64) = [generate_variable_pattern_string(sentence, mentioned_drugs, width) for width in 2:max_width] |> Iterators.flatten |> collect
    
    
    function compute_context_interval(sentence, look_where, context_indices)
        
        sentence_length = lastindex(sentence.words)
        
        @cond begin
            first(context_indices) - first(look_where) >= 1 && last(context_indices) + last(look_where) <= sentence_length => first(context_indices) - first(look_where):1:last(context_indices) + last(look_where)
            first(context_indices) - first(look_where) < 1 && last(context_indices) + last(look_where) <= sentence_length => 1:1:last(context_indices) + last(look_where)
            first(context_indices) - first(look_where) >= 1 && last(context_indices) + last(look_where) > sentence_length => first(context_indices) - first(look_where):1:sentence_length
            first(context_indices) - first(look_where) < 1 && last(context_indices) + last(look_where) > sentence_length => 1:1:sentence_length
        end
    end
    
    
    function compute_context_interval_for_singleton(sentence, look_where, context_indices)
        
        sentence_length = lastindex(sentence.words)
        
        @cond begin
            first(context_indices) - first(look_where) >= 1 && first(context_indices) + last(look_where) <= sentence_length => first(context_indices) - first(look_where):1:first(context_indices) + last(look_where)
            first(context_indices) - first(look_where) < 1 && first(context_indices) + last(look_where) <= sentence_length => 1:1:first(context_indices) + last(look_where)
            first(context_indices) - first(look_where) >= 1 && first(context_indices) + last(look_where) > sentence_length => first(context_indices) - first(look_where):1:sentence_length
            first(context_indices) - first(look_where) < 1 && first(context_indices) + last(look_where) > sentence_length => 1:1:sentence_length
        end
    end 
    
    
    function get_context_sents_for_treatment(document_graph)
        context_sents = []
    
        for index in eachindex(document_graph)
            @match document_graph[index] begin
                1 => push!(context_sents, index)
                _ => nothing
            end
        end
    
        context_sents
    
    end
    
    
    function get_context_sents_for_treatment(document_graph, pattern)
        context_sents = []
    
        for index in eachindex(document_graph)
            @match document_graph[index] begin
                1 => push!(context_sents, index)
                _ => nothing
            end
        end
    
        context_sents, pattern
    
    end
    
    
    
    function generate_interpolation_expression(word, mentioned_drugs, treatment)
    
        drugs_in_treatment = Dict(treatment => "DRUG_$i" for (i, treatment) in enumerate(treatment))
    
        @cond begin
            word.lemma ∈ treatment && word.lemma ∈ getfield.(mentioned_drugs, :name) => drugs_in_treatment[word.lemma]
            word.lemma ∉ treatment && word.lemma ∈ getfield.(mentioned_drugs, :name) => "FOREIGN_DRUG"
            word.lemma ∉ treatment && word.lemma ∉ getfield.(mentioned_drugs, :name) => leave_unchanged_lemma(word)
        end
    
    end
    
    
    function get_literal_context_for_treatment(doc, context_sents_indices, treatment, mentioned_drugs, look_where::Vector{Int64})
    
        context = []
    
        for sentence in doc.sentences[context_sents_indices]

            drug_indices = []

            for word in sentence.words
                if lastindex(treatment) > 1
                    for treatment_part in treatment
                        if word.text == treatment_part
                            push!(drug_indices, word.id)
                        elseif leave_unchanged_lemma(word) == treatment_part
                            push!(drug_indices, word.id)
                        else
                            if occursin(leave_unchanged_lemma(word), treatment_part) && length(leave_unchanged_lemma(word)) > 2
                                push!(drug_indices, word.id)
                            elseif occursin(word.text, treatment_part) && length(word.text) > 2
                                push!(drug_indices, word.id)
                            elseif occursin(treatment_part, word.text) && length(word.text) > 2
                                push!(drug_indices, word.id)
                            else
                                continue
                            end
                        end
                    end
                else
                    if word.text == treatment
                        push!(drug_indices, word.id)
                    elseif leave_unchanged_lemma(word) == treatment
                        push!(drug_indices, word.id)
                    elseif occursin(leave_unchanged_lemma(word), treatment) && length(leave_unchanged_lemma(word)) > 2
                        push!(drug_indices, word.id)
                    elseif occursin(word.text, treatment) && length(word.text) > 2
                        push!(drug_indices, word.id)
                    elseif occursin(treatment, word.text) && length(word.text) > 2
                        push!(drug_indices, word.id)
                    else
                        continue
                    end
                end
            end

            if isempty(drug_indices)
                @show treatment, sentence.text
            end
                        
            start_ind = first(drug_indices)
            end_ind = last(drug_indices)
    
            if start_ind == end_ind
                context_interval = compute_context_interval_for_singleton(sentence, look_where, [start_ind, end_ind])
            else
                context_interval = compute_context_interval(sentence, look_where, [start_ind, end_ind])
            end
    
            context_expression = [generate_interpolation_expression(word, mentioned_drugs, treatment) for word in sentence.words if word.id ∈ context_interval && word.upos ∈ ["NOUN", "PROPN", "PRON", "ADJ", "ADV", "VERB", "NUM", "SYM"]]
            push!(context, context_expression)
        end
    
        context
    
    end
    

    function ordered_character_distance(word, pattern, n)
        num_missing_leq_n = abs(length(word) - length(pattern)) <= n
    
        num_matching_ordered_letters = []
    
        word_chunked = chunk(word, length(word))
        pattern_chunked = chunk(pattern, length(pattern))
    
        if length(word) > length(pattern)
            for (i, pattern_chunk) in enumerate(pattern_chunked)
                if pattern_chunk == word_chunked[i]
                    push!(num_matching_ordered_letters, 1)
                end
            end
        else
            for (i, word_chunk) in enumerate(word_chunked)
                if word_chunk == pattern_chunked[i]
                    push!(num_matching_ordered_letters, 1)
                end
            end
        end
        
        if isempty(num_matching_ordered_letters)
            false
        else
            abs((sum(num_matching_ordered_letters) - max(length(word), length(pattern)) <= n)) && num_missing_leq_n
        end
    
    end
    

    function get_literal_context_for_pattern(doc, patterns_in_context, treatment, mentioned_drugs, look_where::Vector{Int64})
    
        sentence_indices, pattern = patterns_in_context
    
        context = []
    
        pattern_words = @match pattern begin
            ((name1, upos1), (name2, upos2)) => [name1, name2]
            (name, upos) => name
        end
    
        for sentence in doc.sentences[sentence_indices] 
    
            pattern_word_indices = @match pattern_words begin
                [name1, name2] => begin

                    word_ids = []

                    for word in sentence.words
                        if any(word.text .== pattern_words)
                            push!(word_ids, word.id)
                        elseif any(leave_unchanged_lemma(word) .== pattern_words)
                            push!(word_ids, word.id)
                        elseif any(occursin.(pattern_words, leave_unchanged_lemma(word)))
                            push!(word_ids, word.id)
                        end
                    end
                    word_ids
                end
                name => [word.id for word in sentence.words if ((word.text == pattern_words) || (leave_unchanged_lemma(word) == pattern_words) || occursin(pattern_words, leave_unchanged_lemma(word)))]
            end
    
            if first(pattern_word_indices) == last(pattern_word_indices)
                context_interval = compute_context_interval_for_singleton(sentence, look_where, pattern_word_indices)
            else
                context_interval = compute_context_interval(sentence, look_where, pattern_word_indices)
            end
    
            context_expression = [generate_interpolation_expression(word, mentioned_drugs, treatment) for word in sentence.words if word.id ∈ context_interval && word.upos ∈ ["NOUN", "PROPN", "PRON", "ADJ", "ADV", "VERB", "NUM", "SYM"]]
    
            if !isempty(context_expression)
                push!(context, context_expression)
            else
                continue
            end
            
        end
    
        if !isempty(context)
            context
        end
    
    end
    
    
    function get_literal_context_for_pattern(doc, patterns_in_context, treatment, mentioned_drugs, look_where::Vector{Int64})
    
        sentence_indices, pattern = patterns_in_context
    
        context = []
    
        pattern_words = @match pattern begin
            ((name1, upos1), (name2, upos2)) => [name1, name2]
            (name, upos) => name
        end
    
        for sentence in doc.sentences[sentence_indices]           

            pattern_word_indices = @match pattern_words begin
                [name1, name2] => begin

                    word_ids = []

                    for word in sentence.words
                        if any(word.text .== pattern_words)
                            push!(word_ids, word.id)
                        elseif any(leave_unchanged_lemma(word) .== pattern_words)
                            push!(word_ids, word.id)
                        elseif any(occursin.(pattern_words, leave_unchanged_lemma(word)))
                            push!(word_ids, word.id)
                        end
                    end
                    word_ids
                end
                name => [word.id for word in sentence.words if ((word.text == pattern_words) || (leave_unchanged_lemma(word) == pattern_words) || occursin(pattern_words, leave_unchanged_lemma(word)))]
            end
            
            if first(pattern_word_indices) == last(pattern_word_indices)
                context_interval = compute_context_interval_for_singleton(sentence, look_where, pattern_word_indices)
            else
                context_interval = compute_context_interval(sentence, look_where, pattern_word_indices)
            end
    
            context_expression = [generate_interpolation_expression(word, mentioned_drugs, treatment) for word in sentence.words if word.id ∈ context_interval && word.upos ∈ ["NOUN", "PROPN", "PRON", "ADJ", "ADV", "VERB", "NUM", "SYM"]]
    
            push!(context, context_expression)
    
            
        end
    
        if !isempty(context)
            context
        end
    
    end
    
    
    function mix_token(word, weights, rng)
        mix_token = sample(rng, [true, false], Weights(weights), 1)[begin] 
    
        @match mix_token begin
            true => word.upos
            false => leave_unchanged_lemma(word)
        end
    end
    
    
    function generate_typed_interpolation_expression(word, mentioned_drugs, treatment, weights, rng)
    
        drugs_in_treatment = Dict(treatment => "DRUG_$i" for (i, treatment) in enumerate(treatment))
    
        @cond begin
            (word.lemma ∈ treatment) && (word.lemma ∈ getfield.(mentioned_drugs, :name)) => drugs_in_treatment[word.lemma]
            (word.lemma ∉ treatment) && (word.lemma ∈ getfield.(mentioned_drugs, :name)) => "FOREIGN_DRUG"
            (word.lemma ∉ treatment) && (word.lemma ∉ getfield.(mentioned_drugs, :name)) && (word.upos != "NUM") => mix_token(word, weights, rng)
            (word.lemma ∉ treatment) && (word.lemma ∉ getfield.(mentioned_drugs, :name)) && (word.upos == "NUM") => word.upos
        end
    
    end
    
    
    function get_syntactic_type_context_for_pattern(doc, patterns_in_context, treatment, mentioned_drugs, look_where::Vector{Int64}, weights, rng)
    
        sentence_indices, pattern = patterns_in_context
    
        context = []
    
        pattern_words = @match pattern begin
            ((name1, upos1), (name2, upos2)) => [name1, name2]
            (name, upos) => name
        end
    
        for sentence in doc.sentences[sentence_indices] 
    
            pattern_word_indices = @match pattern_words begin
                [name1, name2] => begin

                    word_ids = []

                    for word in sentence.words
                        if any(word.text .== pattern_words)
                            push!(word_ids, word.id)
                        elseif any(leave_unchanged_lemma(word) .== pattern_words)
                            push!(word_ids, word.id)
                        elseif any(occursin.(pattern_words, leave_unchanged_lemma(word)))
                            push!(word_ids, word.id)
                        end
                    end
                    word_ids
                end
                name => [word.id for word in sentence.words if ((word.text == pattern_words) || (leave_unchanged_lemma(word) == pattern_words) || occursin(pattern_words, leave_unchanged_lemma(word)))]
            end
    
            if first(pattern_word_indices) == last(pattern_word_indices)
                context_interval = compute_context_interval_for_singleton(sentence, look_where, pattern_word_indices)
            else
                context_interval = compute_context_interval(sentence, look_where, pattern_word_indices)
            end
    
            context_expression = [generate_typed_interpolation_expression(word, mentioned_drugs, treatment, weights, rng) for word in sentence.words if word.id ∈ context_interval && word.upos ∈ ["NOUN", "PROPN", "PRON", "ADJ", "ADV", "VERB", "SYM", "NUM"]]
    
            push!(context, context_expression)
            
        end
        Iterators.flatten(context) |> collect
    
    end
    
    
    function deep_flatten(collection)
        @match collection begin
            ::Vector{Vector{Any}} => Iterators.flatten(collection) |> collect
            ::Vector{T} where T<:Any => collection
        end
    end
    
    function flatten_expressions(collection)
        @chain collection begin
            Iterators.flatten(_)
            collect
            [deep_flatten(expression_context) for expression_context in _]
        end
    end
    
    instantiate_alias(alias::Tuple{String, String}) = alias[begin]
    instantiate_alias(alias::NTuple{N, Tuple{String, String}} where N) = join([multi_alias[begin] for multi_alias in alias], " ")
    
    function alias_expression(string_expression::String, alias_lookup)
        
        for alias in keys(alias_lookup)
            string_expression = replace(string_expression, alias => alias_lookup[alias])
        end
    
        @chain string_expression begin
            split(_, " ")
            convert(Vector{String}, _)
        end
    
    end
    
    function derive_context_expressions_for_treatment(doc, treatment, mentioned_drugs, look_where::Vector{Int64}, weights, rng)

        document_graph = construct_document_graph_for_string_pattern(doc, treatment)

        # Deriving literal context expressions for treatment
        context_sents_indices_treatment = get_context_sents_for_treatment(document_graph)
        
        literal_context_treatment = get_literal_context_for_treatment(doc, context_sents_indices_treatment, treatment, mentioned_drugs, look_where)
    
        # Initiating search for potential treatment aliases and their respective contexts
        sentence_pointer = findfirst(index -> index == 1, document_graph);
        
        aliases_context = []
    
        aliases = []
    
        # Generate all contexts for aliases
        while true
            all_variable_pattern_candidates = generate_variable_pattern_strings_for_sentence(doc.sentences[sentence_pointer], mentioned_drugs, 2);
    
            push!(aliases, all_variable_pattern_candidates)
            
            document_graphs_for_string_patterns = construct_document_graph_for_string_patterns(doc, all_variable_pattern_candidates);
    
            context_sents_indices = [get_context_sents_for_treatment(document_graph...) for document_graph in document_graphs_for_string_patterns];
            
            literal_contexts = [get_literal_context_for_pattern(doc, context_sents_inds, treatment, mentioned_drugs, look_where) for context_sents_inds in context_sents_indices];
    
            push!(aliases_context, literal_contexts)
    
            # Mixed type context combines literals with syntactic types
            mixed_type_contexts = [get_syntactic_type_context_for_pattern(doc, context_sents_inds, treatment, mentioned_drugs, look_where, weights, rng) for context_sents_inds in context_sents_indices] |> unique;
            push!(aliases_context, mixed_type_contexts)
    
            sentence_pointer = findnext(elem -> elem == 1, document_graph, sentence_pointer + 1)
    
            @cond begin
                isnothing(sentence_pointer) => break;
                !isnothing(sentence_pointer) => continue
            end
        end
    
        context_expressions = literal_context_treatment, aliases_context
    
        final_expressions = []
    
        for expressions in flatten_expressions(context_expressions)
            if typeof(expressions) == Vector{Vector{String}}
                push!(final_expressions, deep_flatten(expressions))
            elseif typeof(expressions) == Vector{String}
                push!(final_expressions, [expressions])
            end
        end
    
        context_expressions = Iterators.flatten(final_expressions) |> collect |> unique
    
        aliases_flattened = Iterators.flatten(aliases) |> collect
    
        string_aliases = [instantiate_alias(alias) for alias in aliases_flattened]
    
        aliases_lookup = Dict(alias => "ALIAS_$i" for (i, alias) in enumerate(string_aliases))
    
        alias_expressions = [alias_expression(join(expression, " "), aliases_lookup) for expression in context_expressions] |> unique
        
        Iterators.filter(elem -> length(elem) > 2, alias_expressions) |> collect
    
    end
    
    
    @doc """
        generate_context_features(doc, learner_params, rng)::Vector{DrugInteractions}
    
    Generate context features from ``doc``. ``learner_params`` are passed to control
    the generation of features. ``rng`` needs to be passed for reproducibility.
    """
    function generate_context_features(doc, learner_params, rng)::Vector{DrugInteraction}
    
        # Identify treatments
        treatments = identify_all_treatments_in_report(doc.parsed_context, doc.mentioned_drugs)
    
        drug_interactions = DrugInteraction[]
    
        for treatment in treatments
            context_expressions = derive_context_expressions_for_treatment(doc.parsed_context, treatment, doc.mentioned_drugs, learner_params.look_where, learner_params.token_mixing_rate, rng)
            push!(drug_interactions, DrugInteraction(Drug.(treatment), InteractionContext(context_expressions)))
        end
    
        drug_interactions
    end

    @doc """
        generate_context_features(doc, treatment::Vector{Drug}, learner_params, rng)::Vector{DrugInteractions}
    
    Generate context features from ``doc``. ``Learner_params`` are passed to control
    the generation of features. ``rng`` needs to be passed for reproducibility.
    """
    function generate_context_features(doc, treatment::Vector{Drug}, learner_params, rng)::Vector{DrugInteraction}
        context_expressions = derive_context_expressions_for_treatment(doc.parsed_context, getfield.(treatment, :name), doc.mentioned_drugs, learner_params.look_where, learner_params.token_mixing_rate, rng)
        DrugInteraction[DrugInteraction(treatment, InteractionContext(context_expressions))]
    end

end