include(joinpath(dirname(@__FILE__), "..", "src", "AtanhErf.jl"))

if !success(`which lockfile`)
    warn("could not find the `lockfile` program. Try installing it (it's included with `procmail` – www.procmail.org).")
end

