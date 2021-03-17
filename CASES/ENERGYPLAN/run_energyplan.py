import subprocess as sp
import os


def create_new_input_file(input_file, new_input_file, x):

    el_demand, COP, effH2, chpeleff, chptheff = x

    with open(input_file, 'r') as f:
        lines = f.readlines()

    skip_line = False
    with open(new_input_file, 'w') as f:
        for i, line in enumerate(lines):
            if not skip_line:
                if str(line) == 'Input_el_demand_Twh=\n':
                    f.write(line)
                    f.write(str(el_demand) + '\n')
                    skip_line = True
                elif str(line) == 'input_eff_hp2_cop=\n':
                    f.write(line)
                    f.write(str(COP) + '\n')
                    skip_line = True
                elif str(line) == 'input_eff_ELTtrans_fuel=\n':
                    f.write(line)
                    f.write(str(effH2) + '\n')
                    skip_line = True
                elif str(line) == 'input_eff_chp2_el=\n':
                    f.write(line)
                    f.write(str(chpeleff) + '\n')
                    skip_line = True
                elif str(line) == 'input_eff_chp3_el=\n':
                    f.write(line)
                    f.write(str(chpeleff) + '\n')
                    skip_line = True
                elif str(line) == 'input_eff_chp2_th=\n':
                    f.write(line)
                    f.write(str(chptheff) + '\n')
                    skip_line = True
                elif str(line) == 'input_eff_chp3_th=\n':
                    f.write(line)
                    f.write(str(chptheff) + '\n')
                    skip_line = True
                else:
                    f.write(line)
            else:
                skip_line = False


def read_output_file(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()

    co2_str = ''
    fuel_str = ''
    for line in lines:
        if 'CO2-emission (total)' in line:
            for j in line[40:46]:
                if j == ',':
                    j = '.'
                co2_str += j
                co2 = float(co2_str)
        if 'Fuel Consumption (total)' in line:
            for j in line[40:46]:
                if j == ',':
                    j = '.'
                fuel_str += j
                fuel = float(fuel_str)

    return co2, fuel


def energyplan(x_tup):

    x = x_tup[1]
    index = x_tup[0]
    path = os.path.dirname(os.path.abspath(__file__))

    ep_path = r'C:\energyPLAN\energyPLAN.exe'
    input_file = os.path.join(path, 'case.txt')

    new_input_file = '%s_%i.txt' % (input_file[:-4], index)

    create_new_input_file(input_file, new_input_file, x)

    result_file = os.path.join(path, 'result_%i.txt' % index)
    cm = [ep_path, '-i', new_input_file, '-ascii', result_file]
    sp.call(cm)

    co2, fuel = read_output_file(result_file)

    os.remove(result_file)
    os.remove(new_input_file)

    return co2, fuel
