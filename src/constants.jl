
#   This file is part of Adapode.jl. It is licensed under the GPL license
#   Adapode Copyright (C) 2019 Michael Reed

const CB = SVector( # Fixed
    SVector( # Euler
        SVector{0,Float64}(),
        SVector(1)),
    SVector( # midpoint
        SVector(0.5),
        SVector(0,1)),
    SVector( # Kutta
        SVector(0.5),
        SVector(-1,2),
        SVector((1,4,1)./6)),
    SVector( # Runge-Kutta
        SVector(0.5),
        SVector(0,0.5),
        SVector(0,0,1),
        SVector((1,2,2,1)./6)))

constants(a,b,c) = SVector(a...,b,b-c)

const CBA = SVector( # Adaptive
    constants(SVector( # Heun-Euler
            SVector(1)),
        SVector(1,0),
        SVector(0.5,0.5)),
    constants(SVector( # Bogacki-Shampine
            SVector(0.5),
            SVector(0,0.75),
            SVector(2/9,1/3,4/9)),
        SVector(7/24, 1/4, 1/3, 1/8),
        SVector(2/9,1/3,4/9,0)),
    constants(SVector( # Fehlberg
            SVector(0.25),
            SVector(0.09375,0.28125),
            SVector(1932/2197,-7200/2197,7296/2197),
            SVector(439/216,-8,3680/513,-845/4104),
            SVector(-8/27,2,3544/2565,1859/4104,-11/40)),
        SVector(16/135,0,6656/12825,28561/56430,-0.18,2/55),
        SVector(25/216,0,1408/2565,2197/4104,-0.2,0)),
    constants(SVector( # Cash-Karp
            SVector(0.2),
            SVector(3/40,9/40),
            SVector(3/40,-9/10,6/5),
            SVector(-11/54,5/2,-70/27,35/27),
            SVector(1631/55296,175/512,575/13824,44275/110592,253/4096)),
        SVector(2825/27648,0,18575/48384,13525/55296,277/14336,0.25),
        SVector(37/378,0,250/621,125/594,0,512/1771)),
    constants(SVector( # Dormand-Prince
            SVector(0.2),
            SVector(3/40,9/40),
            SVector(44/45,-56/15,32/9),
            SVector(19372/6561,-25360/2187,64448/6561,-212/729),
            SVector(9017/3168,-355/33,46732/5247,49/176,-5103/18656),
            SVector(35/384,0,500/1113,125/192,-2187/6784,11/84)),
        SVector(35/384,0,500/1113,125/192,-2187/6784,11/84,0),
        SVector(5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40)))

const CAB = SVector(SVector(1), # Adams-Bashforth
    SVector(-1,3)./2,
    SVector(5,-16,23)./12,
    SVector(-9,37,-59,55)./24,
    SVector(251,-1274,2616,-2774,1901)./720)

const CAM = SVector(SVector(1), # Adams-Moulton
    SVector(1,1)./2,
    SVector(-1,8,5)./12,
    SVector(1,-5,19,9)./24,
    SVector(-19,106,-264,646,251)./720)
