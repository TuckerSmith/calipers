# Commit everything in this folder and push to GitHub. Usage (PowerShell, inside the calipers folder):
#   .\sync_to_github.ps1 "message describing the change"
param([string]$Message = "update from Claude session")
$ErrorActionPreference = "Continue"
# Claude's remote tools cannot write into .github\workflows, so a new CI file arrives at the root:
if (Test-Path "github-ci-workflow.yml") {
    New-Item -ItemType Directory -Force -Path ".github\workflows" | Out-Null
    Move-Item -Force "github-ci-workflow.yml" ".github\workflows\ci.yml"
}
if (Test-Path "push_to_github.ps1") { Remove-Item "push_to_github.ps1" }   # superseded by this script
git add -A
git -c user.name="Tucker" -c user.email="tuckers7401@gmail.com" commit -m $Message
git push origin main
