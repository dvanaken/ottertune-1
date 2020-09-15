#!/bin/bash

if tail -n 1 /etc/hosts | grep -q "localdomain"; then
    echo "/etc/hosts already updated"
else
    tail -n 1 /etc/hosts | awk '{printf "%s %s.localdomain %s\n", $1, $2, $2}' >> /etc/hosts && /etc/init.d/sendmail restart
    echo "Updated /etc/hosts"
fi
