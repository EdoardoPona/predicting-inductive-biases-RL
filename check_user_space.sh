#!/bin/bash

# Ensure the script is run as root to be able to access all user directories
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root"
    exit 1
fi

# Header
echo -e "Username\tSpace Used"

# Get the list of user directories in /home
for userdir in /home/*; do
    # Extract the username from the directory path
    username=$(basename $userdir)

    # Use 'du' to calculate space used by each user and print it
    space_used=$(du -sh $userdir | cut -f1)
    
    echo -e "$username\t$space_used"
done
