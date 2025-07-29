def generate_spice_netlist(
        vdd: float,
        width: float,
        device_name: str,
        m: int,
        nf: int,
        subckt: str,
        corner: str,
        length: float,
        sweep_start: float,
        sweep_stop: float,
        sweep_step: float,
        freq: float = 100e3,
        model_file: str = "n_1p8v_compact_example.l",
        output_file: str = "generated_netlist.sp"
) -> str:
    """
    Generate SPICE netlist files

    parameter:
    vdd          - Power supply voltage (V)
    width        - Transistor width (μm)
    length       - Transistor length (μm)
    sweep_start  - Voltage scan start value (V)
    sweep_stop   - Voltage scan end value (V)
    sweep_step   - Scan step size (V)
    freq         - AC analysis frequency (Hz) Default 100 kHz
    model_file   - Process model file name
    output_file  - Output netlist file name
    """

    netlist_template = f"""
.option nomod tnom=25 post=2 ingold=2 numdgt=7 probe measfile=1 measform=3
.option dccap=1 post
.param vdd={vdd} pi=3.1415926 freq={freq}

{subckt} 3 1 3 2 {device_name} w={width}u l={length}u m={m} nf={nf}

V2 2 0 0
V1 1 0 0 dc=vdd ac=0.05
V3 3 0 0

.lib '{model_file}' {corner}


.op


.ac poi 1 {freq} sweep V1 {sweep_start} {sweep_stop} {sweep_step}


.measure ac cgc param='(ii(V3)/0.05/(2*pi*freq))'

.end
"""

    with open(output_file, 'w') as f:
        f.write(netlist_template)
    return netlist_template


if __name__ == "__main__":
    generate_spice_netlist(
        vdd=1.2,
        width=20,
        m=2,
        nf=2,
        corner='tt',
        subckt='xM1',
        device_name='nmos_main',
        length=10,
        sweep_start=-2,
        sweep_stop=2,
        sweep_step=0.05,
        output_file="template.sp"
    )