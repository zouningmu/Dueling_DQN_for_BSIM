
.option nomod tnom=25 post=2 ingold=2 numdgt=7 probe measfile=1 measform=3
.option dccap=1 post
.param vdd=1.2 pi=3.1415926 freq=100000.0

xM1 3 1 3 2 nmos_main w=20u l=10u m=2 nf=2

V2 2 0 0
V1 1 0 0 dc=vdd ac=0.05
V3 3 0 0

.lib 'n_1p8v_compact_example.l' tt


.op


.ac poi 1 100000.0 sweep V1 -2 2 0.05


.measure ac cgc param='(ii(V3)/0.05/(2*pi*freq))'

.end
