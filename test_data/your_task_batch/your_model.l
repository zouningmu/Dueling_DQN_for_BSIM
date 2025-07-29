*****

.model nmos_main nmos
**************************************************************
*               model flag parameters
**************************************************************
+ level = 54.0                   version = 4.62                 binunit = 2.0
+ paramchk = 1.0                 mobmod = 0                     capmod = 2.0
+ rdsmod = 0                     igcmod = 0                     igbmod = 0
+ rbodymod = 0                   trnqsmod = 0                   acnqsmod = 0
+ diomod = 2.0                   tempmod = 0                    permod = 1.0
+ geomod = 0                     rgeomod = 1.0                  rgatemod = 0
+ wpemod = 1                     fnoimod = 1.0                  tnoimod = 0
+ xpart = 1.0
**************************************************************
*               temperature parameters
**************************************************************
+ tnom = 25.0                    toxe = 4.86E-9                 dtox = 4.43897575939686e-10
+ epsrox = 3.9
+ wint = 2e-8                    lint = -1e-8                   ll = 2.8e-15
+ wl = 4e-15                     lln = 1                        wln = 1
+ lw = 2e-15                     ww = -4.5e-15                  lwn = 1
+ wwn = 1                        lwl = -3e-22                   wwl = -8e-22
+ llc = 0.0                      wlc = 0.0                      lwc = -1e-15
+ wwc = 0                        lwlc = 0                       wwlc = 0.0
+ xl = 2.8E-8                                                   xw = 3.0E-8                                                   dlc = 4.3e-8
+ dwc = 4e-9                     dlcig = 0.0                    dwj = 0
+ lmin = 1.53e-07                lmax = 4.2528e-05              wmin = 2.04e-007
+ wmax = 4.25e-04
**************************************************************
*               dc parameters
**************************************************************
+ vth0 = 0.545                   lvth0 = 0                      wvth0 = 0
+ pvth0 = 1.3E-15                                               k1 = 0.43                      k2 = 0.011
+ wk2 = -6e-9                    lk2 = -4e-9                    pk2 = 8e-16
+ k3 = -8                        k3b = 0
+ w0 = 5e-8                      dvt0 = 2.7                     dvt1 = 0.5
+ dvt2 = 0                       dvt0w = 0                      dvt1w = 5300000.0
+ dvt2w = -0.032                 dsub = 0.4                     ldsub = 0                      pdsub = 0
+ minv = -0.55                   lminv = 0.02e-6                pminv = -5e-15
+ voffl = -3.5e-9                dvtp0 = 4e-7                   dvtp1 = 0
+ lpe0 = 5e-8                    wlpe0 = -2.5e-15               llpe0 = 2e-15                  plpe0 = 4e-22                  lpeb = 0.0                     ados = 1
+ bdos = 1                       xj = 1.4e-7                    ngate = 2.998978685993947e+20
+ ndep = 3e+017                  nsd = 1e+020                   phin = 0.09232076360531513
+ cdsc = 0.003                   ud = 0                         up = 0
+ lp = 1e-008                    cdscb = -0.001                 cdscd = 0                      cit = 0.0
+ voff = -0.145                                                 lvoff = 0
+ nfactor = 1.0                  wnfactor = -1.0E-7
+ lnfactor = 3.5E-8
+ eta0 = 0.1                     leta0 = 3e-9                   etab = -0.1                    wetab = 8e-9
+ u0 = 0.0313                    wu0 = -1.1E-9                  lu0 = 3.5E-10                  pu0 = 1.0E-16
+ ua = -1.6e-9                   wua = 1.5e-16                  lua = -4e-17                   pua = -2e-23
+ ub = 3.3e-18                   wub = -1.5e-25                 lub = -1e-25                   pub = 1.5e-32
+ uc = 1.1e-10                   wuc = 1e-18                    luc = -8e-18                   puc = -5e-25
+ vsat = 100000.0                 wvsat = 0                      lvsat = 0                      pvsat = 3.0E-10
+ a0 = 1.5                       wa0 = 0                        la0 = -5.0E-7
+ ags = 0.55                     wags = -1.5E-8                                                lags = 5e-7
+ pags = 3e-14                   a1 = 0                         a2 = 1
+ b0 = 0                         lb0 = -3e-14                   pb0 = 8e-21                    b1 = 0
+ keta = -0.0319425              lketa = -2.3e-8                pketa = 8e-16
+ dwg = -2.132395e-9             ldwg = 8.956059e-17
+ dwb = 4.404375e-009            pclm = 0.1                     lpclm = 0
+ pdiblc1 = 0.4                  pdiblc2 = 2e-4                 lpdiblc2 = 2e-9                pdiblcb = 0
+ drout = 0.75                   pvag = 0                       delta = 0.01                   ldelta = 3e-9                  pscbe1 = 5.936e+008
+ pscbe2 = 1.0e-5                fprout = 100                   pdits = 0.1
+ pditsd = 0.3                   pditsl = 2e6
+ rsh = 13.3                     rdsw = 200                     rdswmin = 0
+ prwg = 0                       prwb = 0                       alpha0 = -8e-7
+ alpha1 = 6.7944                palpha1 = -1e-14               beta0 = 17.5856                lbeta0 = -5e-7                 pbeta0 = -3e-14                agidl = 4.0E-8
+ bgidl = 1.5E9                                                 wbgidl = -40.0
+ lbgidl = -5.0                                                 pbgidl = 3.0E-6                                               cgidl = 25
+ wcgidl = 1e-5                  egidl = 0.65                   wegidl = 0
+ aigbacc = 0.43                 bigbacc = 0.054                cigbacc = 0.075
+ nigbacc = 1.0                  aigbinv = 0.35                 bigbinv = 0.03
+ cigbinv = 0.006                eigbinv = 1.1                  nigbinv = 3.0
+ aigc = 0.0101                  bigc = 0.001472
+ lbigc = -9e-006                cigc = 0.075                   aigsd = 0.43
+ bigsd = 0.054                  cigsd = 0.075                  lcigsd = 0.0
+ nigc = 1.0                     poxedge = 1.0                  pigcd = 1.0                    ntox = 1
**************************************************************
*               capacitance parameters
**************************************************************
+ cgso = 1.95E-10                                               cgdo = 1.95E-10                                               cgbo = 0
+ cgdl = 2.25E-11                                               lcgdl = 0                                                     wcgdl = 0
+ pcgdl = 2.0E-24                                               cgsl = 2.25E-11                                               lcgsl = 0
+ wcgsl = 0                      pcgsl = 2.0E-24                                               clc = 1.0e-7
+ cle = 0.6                      cf = 8.22069e-11                                              wcf = 0
+ pcf = 0                        ckappas = 5                    lckappas = 0
+ wckappas = 0                   pckappas = 0                   ckappad = 5
+ lckappad = 0                   wckappad = 0                   pckappad = 0
+ acde = 0.6735024853017725                    moin = 5.0                       noff = 2.0
+ lnoff = 1.5e-7                 wnoff = 1e-7
+ voffcv = -0.10813445533506619                 wvoffcv = -1e-8                lvoffcv = -2e-8                pvoffcv = 5e-15                ute = -1.05                    wute = 1.05e-7
+ lute = 4e-8                    kt1 = -0.18
+ wkt1 = 8e-9                    pkt1 = 2e-16                   kt1l = -7e-9
+ kt2 = -0.03                    ua1 = 2.8e-9                   lua1 = -1.3e-16                ub1 = -3e-18                   wub1 = 2.2e-25
+ lub1 = 2.5e-25                 uc1 = -7.0e-11                 wuc1 = 2e-17
+ luc1 = 1e-17                   at = 2.5e4                     pat = -3.5e-10                 prt = 0                        tvoff = 0.002
**************************************************************
*               noise parameters
**************************************************************
+ em = 1e6
+ ef = 0.9                       lintnoi = 1e-8
+ noia = 3.0E41                  noib = 1.0E24                                                 noic = 1.0E8
**************************************************************
*               diode parameters
**************************************************************
+ jss = 2.300000e-07             jsws = 1.188000e-13            jswgs = 6.08e-14
+ njs = 1.04                     ijthsfwd = 0.00125             ijthsrev = 0.001
+ bvs = 10.7                     xjbvs = 1                      jtss = 0
+ jtssws = 0                     jtsswgs = 2.83658e-09                                         njts = 20
+ njtssw = 20                    njtsswg = 12                   xtss = 0.02
+ xtsd = 0.02                    xtssws = 0.02                  xtsswd = 0.02
+ xtsswgs = 0.3                  xtsswgd = 0.3                  tnjts = 0
+ tnjtssw = 0                    tnjtsswg = 0                   vtsswgs = 2
+ pbs = 9.600000e-01             cjs = 0.0010535                                               mjs = 4.720000e-01
+ pbsws = 4.660000e-01           cjsws = 7.9E-11                                               mjsws = 3.206000e-01
+ pbswgs = 0.696                 cjswgs = 4.0E-10                                              mjswgs = 0.3306
+ tpb = 1.340000e-03             tcj = 8.920000e-04             tpbsw = 1.150000e-03
+ tcjsw = 8.400000e-04           tpbswg = 0.00165               tcjswg = 6.8e-4
+ xtis = 3.000000e+00
**************************************************************
*               layout related parameters
**************************************************************
+ dmcg = 1.9975e-7               dmci = 3.9525e-7
**************************************************************
*               rf parameters
**************************************************************
**************************************************************
*               stress parameters
**************************************************************
+ web = 1                        wec = 1                        kvth0we = 0.003
+ wkvth0we = -2e-10              lkvth0we = 1.2e-9              pkvth0we = -2.5e-16
+ k2we = 0                       ku0we = 0
+ lku0we = -2e-9                 pku0we = 4e-16                 scref = 1.7e-6
+ saref = 0.595e-6               sbref = 0.595e-6               wlod = 0
+ ku0 = -2e-8                    kvsat = 0.5                    kvth0 = 1e-9
+ tku0 = 0                       llodku0 = 1.0                  wlodku0 = 1.0
+ llodvth = 1.0                  wlodvth = 1.0                  lku0 = -1e-7
+ wku0 = 1e-7                    pku0 = 0                       lkvth0 = -1.4e-7
+ wkvth0 = -1e-7                 pkvth0 = 8e-14                 stk2 = 0
+ lodk2 = 1.0                    steta0 = 0                     lodeta0 = 1.0
****************************************************************
*                           soa
****************************************************************
+ rshg = 14.4
+ vgs_max = 1.98                 vgd_max = 1.98
+ vds_max = 1.98                 vbd_max = 3.63
+ vbs_max = 3.63                 vgb_max = 3.63



