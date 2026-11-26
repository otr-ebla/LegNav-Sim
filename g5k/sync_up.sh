#!/bin/bash
SITE=${SITE:-nancy}
rsync -avuP --exclude .git --exclude data --exclude figures --exclude legacy --exclude __pycache__ --exclude assets --exclude logs ./ "$SITE.g5k:~/legnav/"
