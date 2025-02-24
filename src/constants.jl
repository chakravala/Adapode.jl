
#   This file is part of Adapode.jl. It is licensed under the GPL license
#   Adapode Copyright (C) 2019 Michael Reed

const CB = Values( # Fixed
    Values( # Euler
        Values{0,Float64}(),
        Values(1)),
    Values( # midpoint
        Values(1/2),
        Values(0,1)),
    Values( # Kutta
        Values(1/2),
        Values(-1,2),
        Values((1,4,1)./6)),
    Values( # Runge-Kutta
        Values(0.5),
        Values(0,0.5),
        Values(0,0,1),
        Values((1,2,2,1)./6)))

constants(a,b,c) = Values(a...,b,b-c)

const CBA = Values( # Adaptive
    constants(Values( # Heun-Euler
            Values(1)),
        Values(1,0),
        Values(1,1)./2),
    constants(Values( # Bogacki-Shampine
            Values(1/2),
            Values(0,3/4),
            Values(2,3,4)./9),
        Values(7/24, 1/4, 1/3, 1/8),
        Values(2/9,1/3,4/9,0)),
    constants(Values( # Fehlberg
            Values(1/4),
            Values(0.09375,0.28125),
            Values(1932/2197,-7200/2197,7296/2197),
            Values(439/216,-8,3680/513,-845/4104),
            Values(-8/27,2,3544/2565,1859/4104,-11/40)),
        Values(16/135,0,6656/12825,28561/56430,-0.18,2/55),
        Values(25/216,0,1408/2565,2197/4104,-0.2,0)),
    constants(Values( # Cash-Karp
            Values(1/5),
            Values(3/40,9/40),
            Values(3/40,-9/10,6/5),
            Values(-11/54,5/2,-70/27,35/27),
            Values(1631/55296,175/512,575/13824,44275/110592,253/4096)),
        Values(2825/27648,0,18575/48384,13525/55296,277/14336,0.25),
        Values(37/378,0,250/621,125/594,0,512/1771)),
    constants(Values( # Dormand-Prince
            Values(1/5),
            Values(3/40,9/40),
            Values(44/45,-56/15,32/9),
            Values(19372/6561,-25360/2187,64448/6561,-212/729),
            Values(9017/3168,-355/33,46732/5247,49/176,-5103/18656),
            Values(35/384,0,500/1113,125/192,-2187/6784,11/84)),
        Values(35/384,0,500/1113,125/192,-2187/6784,11/84,0),
        Values(5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40)))

const CAB = Values(Values(1), # Adams-Bashforth
    Values(-1,3)./2,
    Values(5,-16,23)./12,
    Values(-9,37,-59,55)./24,
    Values(251,-1274,2616,-2774,1901)./720)

const CAM = Values(Values(1), # Adams-Moulton
    Values(1,1)./2,
    Values(-1,8,5)./12,
    Values(1,-5,19,9)./24,
    Values(-19,106,-264,646,251)./720)

const Gauss = Values(
    Values(
        Values(1),
        Values(
            Values(1,1)/3)),
    Values(
        Values(1,1,1)/3,
        Values(
            Values(1,1)/6,
            Values(4,1)/6,
            Values(1,4)/6)),
    Values(
        Values(-27,25,25,25)/48,
        Values(
            Values(1,1)/3,
            Values(1,1)/5,
            Values(3,1)/5,
            Values(1,3)/5)),
    Values(
        Values(35494641/158896895,35494641/158896895,35494641/158896895,40960013/372527180,40960013/372527180,40960013/372527180),
        Values(
            Values(100320057/224958844,100320057/224958844),
            Values(100320057/224958844,16300311/150784976),
            Values(16300311/150784976,100320057/224958844),
            Values(13196394/144102857,13196394/144102857),
            Values(13196394/144102857,85438943/104595944),
            Values(85438943/104595944,13196394/144102857))))
