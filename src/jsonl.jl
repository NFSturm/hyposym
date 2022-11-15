module JSONL 
    """
    A module for working with JSONL files. Based on PyCall.
    """

    using PyCall

    const jsonlines = PyNULL()

    function __init__() 
        copy!(jsonlines, pyimport("jsonlines"))
    end

    @doc """
        read_jsonl(path)

    Reads a JSONL file from `path` and returns a dictionary.
    """
    function read_jsonl(path) 
        @pywith jsonlines.open(path, "r") as jsonl_file begin
            lst = [obj for obj in jsonl_file]
            return lst
        end
    end
end