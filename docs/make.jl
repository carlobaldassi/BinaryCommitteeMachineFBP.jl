using Documenter, BinaryCommitteeMachineFBP

CIbuild = get(ENV, "CI", nothing) == "true"

makedocs(
    modules  = [BinaryCommitteeMachineFBP],
    format   = Documenter.HTML(prettyurls = CIbuild),
    sitename = "BinaryCommitteeMachineFBP.jl"
    )

deploydocs(
    repo   = "github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl.git",
)
