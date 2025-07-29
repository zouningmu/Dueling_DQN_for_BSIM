
.option nomod tnom=25 post=2 ingold=2 numdgt=7 probe measfile=1 measform=3
.option dccap=1 post 
.param vdd=2.0 pi=3.1415926 freq=100000

M1 3 1 3 3 nmos_main w=20u l=10u m=18 nf=2

V1 1 0 0 dc=vdd ac=0.05

V3 3 0 0 	$ac voltage; cgc's ac voltage location is different from cgg


.include your_model.l

.op

.ac poi 1 freq sweep V1 "-vdd" "vdd" 0.05	$simulation frequency 100K; sweep voltage from -vdd to vdd

.measure ac cgg param=('(ii(V1)/0.05/(2*pi*freq))')	$easy to collect result; C=I(image)/V(ac_voltage)/(2*π*f)

.end
