#!/bin/bash

expect -c "
set timeout 180
spawn sh /root/Anaconda3-2019.03-Linux-x86_64.sh
expect \">>>\"
send \"\n\"
expect {
    \"More\" {
        send \"\n\"
        exp_continue
    }
    \">>>\" {
        send \"yes\n\"
    }
}
expect \">>>\"
send \"\n\"
expect {
    \">>>\" {
        send \"yes\n\"
    }
}
"
