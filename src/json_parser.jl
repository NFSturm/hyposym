module JSONParser 

    using JSON3

    @doc """
        parse_json(filepath::String)

    Parses a JSON file into a specified type.
    """
    function parse_json(filepath::String, target_type::Type = nothing)
        json_string = read(filepath)
        doc = JSON3.read(json_string)
        convert(target_type, doc)
    end

end