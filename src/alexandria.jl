module Alexandria
    """
    A module for specifying data dependencies. Based on TOML.
    """

    export define_dataset, get_dataset, DATA_SPEC_PATH

    using MLStyle
    using Chain
    using TOML
    using ResultTypes
    using UUIDs

    const DATA_SPEC_PATH = "./Data.toml"

    get_existing_datasets(data_path)::Result{Dict, KeyError} = @chain TOML.parsefile(data_path) begin
        get(_, "datasets", KeyError)
    end

    function create_or_append_datasets(new_dataset, existing_datasets)::Dict
        @match isfile("./Data.toml") begin
            false => Dict("datasets" => new_dataset)
            true => Dict("datasets" => merge(new_dataset, existing_datasets))
        end
    end
    
    function define_dataset(dataset_name::String, description::String, path::String, data_spec_path = DATA_SPEC_PATH)
    
        # Extract existing dataset
        maybe_existing_datasets = get_existing_datasets(data_spec_path)
        (; result) = maybe_existing_datasets
        existing_datasets = something(result)

        # Define Dict with new dataset
        new_dataset = Dict(dataset_name => Dict("name" => dataset_name, "description" => description, "path" => path, "uuid" => string(UUIDs.uuid4())))
    
        # Create if there are no datasets yet, append if there are entries already
        datasets = create_or_append_datasets(new_dataset, existing_datasets)
    
        # Write to file in default path
        open(joinpath(data_spec_path), "w") do io
            TOML.print(
                io, 
                datasets
            )
        end
    end

    function get_dataset(dataset_name::String)
        data_specs = TOML.parsefile("./Data.toml")
        data_specs["datasets"][dataset_name]
    end

    # function delete_dataset()

    # end


end