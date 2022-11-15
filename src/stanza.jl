module Stanza 
    """
    A module for working with the Python NLP library Stanza. Based on PyCall.
    """

    using PyCall

    const stanza = PyNULL()
    const nlp = PyNULL()

    function __init__() 
        copy!(stanza, pyimport("stanza"))
        nlp_pipe = stanza.Pipeline("en")
        copy!(nlp, nlp_pipe)
    end
end