# One-time push of this folder to GitHub. Run from PowerShell inside the calipers folder:
#   cd "$env:USERPROFILE\Documents\3D Printing\calipers"; .\push_to_github.ps1
# Prerequisites: git installed, an EMPTY repo named "calipers" created under your GitHub account
# (no README), and you are signed in to GitHub in git (Git Credential Manager will prompt).
$ErrorActionPreference = "Stop"
# The CI workflow could not be written into .github\workflows by the remote tools (protected path);
# move it into place here if it is still at the root.
if ((Test-Path "github-ci-workflow.yml") -and -not (Test-Path ".github\workflows\ci.yml")) {
    New-Item -ItemType Directory -Force -Path ".github\workflows" | Out-Null
    Move-Item "github-ci-workflow.yml" ".github\workflows\ci.yml"
}
$remote = "https://github.com/tuckers7401/calipers.git"
if (-not (Test-Path ".git")) {
    git init -b main
    git add -A
    git -c user.name="Tucker" -c user.email="tuckers7401@gmail.com" commit -m "calipers 0.1.0 - Phase 1: the digital calipers

Geometry core (mesh + B-rep), measurement primitives, feature extraction,
report/digest, headless rendering, export, CLI; 49 tests against a
ground-truth bracket, the NIST AM test artifact and the 3DBenchy.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018Vt2o3vekz7gA8upLuYUNu"
}
git remote remove origin 2>$null
git remote add origin $remote
git push -u origin main
Write-Host ""
Write-Host "Pushed. Next: install the Claude GitHub App on the repo (github.com/apps/claude) so cloud sessions can push."
