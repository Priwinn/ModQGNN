# Generated by OpenQL 0.10.0 for program my_program
version 1.2

pragma @ql.name("my_program")


.my_kernel
    x45 q[2]
    { # start at cycle 1
        x q[3]
        y90 q[2]
    }
    cz q[3], q[2]
    ym90 q[0]
    ym90 q[0]
    cz q[2], q[0]
    skip 2
    { # start at cycle 8
        y90 q[0]
        ym90 q[2]
    }
    cz q[0], q[2]
    skip 2
    { # start at cycle 12
        y90 q[2]
        ym90 q[0]
    }
    cz q[2], q[0]
    skip 1
    y45 q[4]
    cz q[2], q[4]
    skip 2
    { # start at cycle 19
        x45 q[1]
        x45 q[2]
    }
    cz q[2], q[1]
    skip 2
    xm90 q[2]
    cz q[3], q[2]
    skip 1
    ym90 q[1]
    cz q[2], q[1]
    skip 2
    { # start at cycle 30
        y90 q[1]
        ym90 q[2]
    }
    cz q[1], q[2]
    skip 2
    { # start at cycle 34
        y90 q[2]
        ym90 q[1]
    }
    cz q[2], q[1]
    skip 2
    cz q[4], q[2]
    skip 2
    { # start at cycle 41
        y90 q[0]
        y90 q[1]
        xm90 q[2]
        x90 q[4]
    }
