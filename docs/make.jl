using Documenter, BinaryCommitteeMachineFBP

makedocs()

deploydocs(
    deps   = Deps.pip("pygments", "mkdocs", "mkdocs-bootswatch", "python-markdown-math"),
    repo   = "github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl.git",
    julia  = "0.5"
)

