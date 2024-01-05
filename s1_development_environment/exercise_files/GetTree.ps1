# Customized PowerShell script to display a directory tree including only specific directories and their files

$targetDirectories = @("final_exercise")

function Get-SelectiveTree {
    param (
        [string]$path = (Get-Location),
        [int]$indentLevel = 0
    )

    # Get the current directory
    $currentDir = Get-Item -LiteralPath $path

    # Check if the current directory is in the list of target directories
    if ($targetDirectories -contains $currentDir.Name) {
        # Display the current directory with proper indentation
        Write-Output (" " * $indentLevel * 4) + $currentDir.Name

        # Get all items (files and subdirectories) in the current directory
        $items = Get-ChildItem -LiteralPath $currentDir.FullName

        # Recursively process subdirectories
        foreach ($item in $items) {
            if ($item.PSIsContainer) {
                Get-SelectiveTree -path $item.FullName -indentLevel ($indentLevel + 1)
            } else {
                # Display files with proper indentation
                Write-Output (" " * ($indentLevel + 1) * 4) + $item.Name
            }
        }
    }
}