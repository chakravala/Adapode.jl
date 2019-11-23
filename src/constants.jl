
#   This file is part of Adapode.jl. It is licensed under the GPL license
#   Adapode Copyright (C) 2019 Michael Reed

function midpoint2nd()
    2,SMatrix{2,2,Float64}([0 0; 0.5 0]),SVector(0,1)
end

function kutta3rd()
    3,SMatrix{3,3,Float64}([0 0 0; 0.5 0 0; -1 2 0]),SVector((1,4,1)./6)
end

function classical4th()
    4,SMatrix{4,4,Float64}([0 0 0 0; 0.5 0 0 0; 0 0.5 0 0; 0 0 1 0]),SVector((1,2,2,1)./6)
end

function adaptive_heun_euler()
    c = SVector(1,0)
    2,SMatrix{2,2,Float64}([0 0; 1 0]),c,SVector(c-SVector(0.5,0.5))
end

function bogacki_shampine_rk23()
    c =  SVector(7/24, 1/4, 1/3, 1/8)
    4,SMatrix{4,4,Float64}([0 0 0 0; 0.5 0 0 0; 0 0.75 0 0;2/9 1/3 4/9 0]),c,c-SVector(2/9,1/3,4/9,0.0)
end

function fehlberg_rk45()
    c = SVector(16/135,0,6656/12825,28561/56430,-0.18,2/55)
    brs = SMatrix{6,6,Float64}([0 0 0 0 0 0; 0.25 0 0 0 0 0; 0.09375 0.28125 0 0 0 0;
        1932/2197 -7200/2197 7296/2197 0 0 0; 439/216 -8 3680/513 -845/4104 0 0; -8/27 2 3544/2565 1859/4104 -11/40 0])
    6,brs,c,c-SVector(25/216,0,1408/2565,2197/4104,-0.2,0)
end

function cash_karp_rk45()
    c = SVector(2825/27648,0,18575/48384,13525/55296,277/14336,0.25)
    brs = SMatrix{6,6,Float64}([0 0 0 0 0 0; 0.2 0 0 0 0 0; 3/40 9/40 0 0 0 0;
        3/40 -9/10 6/5 0 0 0; -11/54 5/2 -70/27 35/27 0 0; 1631/55296 175/512 575/13824 44275/110592 253/4096 0])
    6,brs,c,c-SVector(37/378,0,250/621,125/594,0,512/1771)
end

function dormand_prince_rk45()
    c = SVector(35/384,0,500/1113,125/192,-2187/6784,11/84,0)
    brs = SMatrix{7,7,Float64}([0 0 0 0 0 0 0; 0.2 0 0 0 0 0 0; 3/40 9/40 0 0 0 0 0
        44/45 -56/15 32/9 0 0 0 0; 19372/6561 -25360/2187 64448/6561 -212/729 0 0 0
        9017/3168 -355/33 46732/5247 49/176 -5103/18656 0 0; 35/384 0 500/1113 125/192 -2187/6784 11/84 0])
    7,brs,c,c-SVector(5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40)
end

function abm2nd(eqc)
    2,SVector((3,-1)./2),SVector((1,1)./2),Vector(undef,3)
end

function abm3rd(eqc)
    3,SVector((23,-16,5)./12),SVector((5,8,-1)./12),Vector(undef,4)
end

function abm4th(eqc)
    4,SVector((55,-59,37,-9)./24),SVector((9,19,-5,1)./24),Vector(undef,5)
end

function abm5th(eqc)
    5,SVector((1901,-2774,2616,-1274,251)./720),SVector((251,646,-264,106,-19)./720),Vector(undef,6)
end
