git config user.name "mohitmasters1984"
git config user.email "mohitmasters1984@gmail.com"
param(
    [string]$RemoteUrl = "",
    [switch]$Push
)

# Show commands before running
function Show-And-Run([string]$cmd) {
    Write-Host "Running: $cmd"
    iex $cmd
}

# Configure git user
Show-And-Run "git config user.name \"mohitmasters1984\""
Show-And-Run "git config user.email \"mohitmasters1984@gmail.com\""

# Initialize repo if needed
if (-not (Test-Path .git)) {
    Show-And-Run "git init"
    Show-And-Run "git branch -M main"
    Write-Host "Initialized new git repository and set branch to 'main'."
} else {
    Write-Host ".git already exists — using existing repository."
}

# Stage the frontend change (safe: only this file)
$target = "frontend/js/api.js"
if (Test-Path $target) {
    Show-And-Run "git add $target"
    $status = git status --porcelain
    if ($status) {
        Show-And-Run "git commit -m \"Make frontend API base URL configurable\""
        Write-Host "Committed changes to $target"
    } else {
        Write-Host "No changes to commit for $target"
    }
} else {
    Write-Host "WARNING: $target not found. Nothing was staged."
}

# Optionally set remote
if ($RemoteUrl -ne "") {
    # show exact commands
    Show-And-Run "git remote remove origin 2>$null | Out-Null"
    Show-And-Run "git remote add origin $RemoteUrl"
    Write-Host "Remote 'origin' set to: $RemoteUrl"
} else {
    Write-Host "No remote provided. To add a remote later:"
    Write-Host "  git remote add origin <your-repo-url>"
}

# If -Push flag provided, require explicit approval before pushing
if ($Push) {
    Write-Host "About to push to remote 'origin' on branch 'main'."
    Write-Host "You must type the exact phrase: Approve Push"
    $approval = Read-Host "Type approval phrase to continue"
    if ($approval -ne 'Approve Push') {
        Write-Host "Approval not provided. Aborting push."
        exit 1
    }

    # Show and run push
    Show-And-Run "git push -u origin main"
    Write-Host "Push completed (or error shown above)."
} else {
    Write-Host "Push not requested. To push now, re-run with -Push and provide approval when prompted."
}
