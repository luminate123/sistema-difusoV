from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

# Definir los antecedentes (inputs)
pain = ctrl.Antecedent(np.arange(0, 11, 0.1), 'pain')
temperature = ctrl.Antecedent(np.arange(36.5, 40.5, 0.1), 'temperature')
hemorrhage = ctrl.Antecedent(np.arange(0, 11, 1), 'hemorrhage')
appetite = ctrl.Antecedent(np.arange(0, 11, 1), 'appetite')
muscle_weakness = ctrl.Antecedent(np.arange(0, 11, 1), 'muscle_weakness')
difficult_breath = ctrl.Antecedent(np.arange(0, 11, 0.1), 'difficult_breath')

# Definir el universo de discurso para el dengue en porcentaje
dengue = ctrl.Consequent(np.arange(0, 101, 1), 'dengue')
# Definir las funciones de membresía
# Dolor (Pain)
pain['No pain'] = fuzz.trimf(pain.universe, [0, 0, 2.5])
pain['Mild'] = fuzz.trimf(pain.universe, [1.5, 2.5, 4.5])
pain['Moderate'] = fuzz.trimf(pain.universe, [3.5, 5, 6.5])
pain['Strong'] = fuzz.trimf(pain.universe, [5.5, 7, 8.5])
pain['Severe'] = fuzz.trapmf(pain.universe, [7.5, 8.5, 10, 10])

# Temperatura (Temperature)
temperature['Normal'] = fuzz.trimf(temperature.universe, [36.5, 36.6, 37.0])
temperature['Low Fever'] = fuzz.trimf(temperature.universe, [37.1, 37.5, 37.9])
temperature['High Fever'] = fuzz.trimf(temperature.universe, [38.0, 39, 40])

# Hemorragia (Hemorrhage)
hemorrhage['Mild'] = fuzz.trimf(hemorrhage.universe, [0, 2, 4])
hemorrhage['Medium'] = fuzz.trimf(hemorrhage.universe, [3, 5, 8])
hemorrhage['Severe'] = fuzz.trimf(hemorrhage.universe, [7, 9, 10])

# Apetito (Appetite)
appetite['Normal'] = fuzz.trimf(appetite.universe, [0, 2, 4.5])
appetite['Little'] = fuzz.trimf(appetite.universe, [3.5, 5.5, 7.5])
appetite['None'] = fuzz.trimf(appetite.universe, [6.5, 8, 10])

# Debilidad muscular (Muscle Weakness)
muscle_weakness['Normal'] = fuzz.trimf(muscle_weakness.universe, [0, 1.5, 3])
muscle_weakness['Mild'] = fuzz.trimf(muscle_weakness.universe, [2.5, 4, 5.5])
muscle_weakness['Moderate'] = fuzz.trimf(muscle_weakness.universe, [4.5, 6.5, 8.5])
muscle_weakness['Severe'] = fuzz.trapmf(muscle_weakness.universe, [7.5, 9, 10, 10])

# Dificultad para respirar (Difficult Breathing)
difficult_breath['Normal'] = fuzz.trimf(difficult_breath.universe, [0, 1, 2])
difficult_breath['Low'] = fuzz.trimf(difficult_breath.universe, [1.5, 3, 4.5])
difficult_breath['Moderate'] = fuzz.trimf(difficult_breath.universe, [4, 5.5, 7])
difficult_breath['Severe Low'] = fuzz.trimf(difficult_breath.universe, [6, 7.5, 9])
difficult_breath['Severe'] = fuzz.trimf(difficult_breath.universe, [8, 9, 10])

# Definir las enfermedades (outputs) en porcentaje
dengue['Mild'] = fuzz.trimf(dengue.universe, [0, 20, 39.9])
dengue['Moderate'] = fuzz.trimf(dengue.universe, [40, 50, 69.9])
dengue['Strong'] = fuzz.trimf(dengue.universe, [70, 85, 100])


# Define the rules (assuming rule1_dengue and rule1_yellow_fever are defined elsewhere)
rule1_dengue = ctrl.Rule(pain['Moderate'] & temperature['Low Fever'] & hemorrhage['Mild'] & 
                         appetite['Little'] & muscle_weakness['Severe'] & difficult_breath['Normal'], 
                         dengue['Strong'])
# Reglas para el diagnóstico de dengue
rule2_dengue = ctrl.Rule(pain['Mild'] & temperature['Normal'] & hemorrhage['Mild'] & 
                         appetite['Normal'] & muscle_weakness['Mild'] & difficult_breath['Normal'], 
                         dengue['Mild'])

rule3_dengue = ctrl.Rule(pain['Strong'] & temperature['High Fever'] & hemorrhage['Medium'] & 
                         appetite['None'] & muscle_weakness['Severe'] & difficult_breath['Moderate'], 
                         dengue['Strong'])

rule4_dengue = ctrl.Rule(pain['No pain'] & temperature['Normal'] & hemorrhage['Mild'] & 
                         appetite['Normal'] & muscle_weakness['Normal'] & difficult_breath['Normal'], 
                         dengue['Mild'])

rule5_dengue = ctrl.Rule(pain['Moderate'] & temperature['High Fever'] & hemorrhage['Medium'] & 
                         appetite['Little'] & muscle_weakness['Moderate'] & difficult_breath['Low'], 
                         dengue['Moderate'])

rule6_dengue = ctrl.Rule(pain['Severe'] & temperature['Low Fever'] & hemorrhage['Severe'] & 
                         appetite['None'] & muscle_weakness['Severe'] & difficult_breath['Severe'], 
                         dengue['Strong'])

rule7_dengue = ctrl.Rule(pain['Mild'] & temperature['Low Fever'] & hemorrhage['Mild'] & 
                         appetite['Little'] & muscle_weakness['Normal'] & difficult_breath['Low'], 
                         dengue['Moderate'])

rule8_dengue = ctrl.Rule(pain['Moderate'] & temperature['Normal'] & hemorrhage['Mild'] & 
                         appetite['Little'] & muscle_weakness['Mild'] & difficult_breath['Moderate'], 
                         dengue['Mild'])

rule9_dengue = ctrl.Rule(pain['Strong'] & temperature['High Fever'] & hemorrhage['Severe'] & 
                         appetite['None'] & muscle_weakness['Severe'] & difficult_breath['Severe'], 
                         dengue['Strong'])

rule10_dengue = ctrl.Rule(pain['Moderate'] & temperature['High Fever'] & hemorrhage['Mild'] & 
                          appetite['Little'] & muscle_weakness['Mild'] & difficult_breath['Low'], 
                          dengue['Moderate'])

rule11_dengue = ctrl.Rule(pain['Severe'] & temperature['Low Fever'] & hemorrhage['Severe'] & 
                          appetite['None'] & muscle_weakness['Moderate'] & difficult_breath['Severe Low'], 
                          dengue['Strong'])

rule12_dengue = ctrl.Rule(pain['No pain'] & temperature['Normal'] & hemorrhage['Mild'] & 
                          appetite['Normal'] & muscle_weakness['Normal'] & difficult_breath['Low'], 
                          dengue['Mild'])

rule13_dengue = ctrl.Rule(pain['Strong'] & temperature['High Fever'] & hemorrhage['Medium'] & 
                          appetite['Little'] & muscle_weakness['Moderate'] & difficult_breath['Moderate'], 
                          dengue['Moderate'])

rule14_dengue = ctrl.Rule(pain['Moderate'] & temperature['High Fever'] & hemorrhage['Severe'] & 
                          appetite['None'] & muscle_weakness['Severe'] & difficult_breath['Severe'], 
                          dengue['Strong'])

rule15_dengue = ctrl.Rule(pain['Mild'] & temperature['Low Fever'] & hemorrhage['Mild'] & 
                          appetite['Normal'] & muscle_weakness['Mild'] & difficult_breath['Low'], 
                          dengue['Mild'])

rule16_dengue = ctrl.Rule(pain['Severe'] & temperature['High Fever'] & hemorrhage['Medium'] & 
                          appetite['None'] & muscle_weakness['Severe'] & difficult_breath['Severe Low'], 
                          dengue['Strong'])

rule17_dengue = ctrl.Rule(pain['Moderate'] & temperature['Low Fever'] & hemorrhage['Medium'] & 
                          appetite['Little'] & muscle_weakness['Moderate'] & difficult_breath['Low'], 
                          dengue['Moderate'])

rule18_dengue = ctrl.Rule(pain['No pain'] & temperature['Normal'] & hemorrhage['Mild'] & 
                          appetite['Little'] & muscle_weakness['Normal'] & difficult_breath['Moderate'], 
                          dengue['Mild'])

rule19_dengue = ctrl.Rule(pain['Strong'] & temperature['High Fever'] & hemorrhage['Severe'] & 
                          appetite['None'] & muscle_weakness['Severe'] & difficult_breath['Severe'], 
                          dengue['Strong'])

rule20_dengue = ctrl.Rule(pain['Mild'] & temperature['Normal'] & hemorrhage['Medium'] & 
                          appetite['Little'] & muscle_weakness['Mild'] & difficult_breath['Moderate'], 
                          dengue['Moderate'])

rule21_dengue = ctrl.Rule(pain['Moderate'] & temperature['Low Fever'] & hemorrhage['Severe'] & 
                          appetite['Little'] & muscle_weakness['Moderate'] & difficult_breath['Severe Low'], 
                          dengue['Strong'])

rule22_dengue = ctrl.Rule(pain['No pain'] & temperature['Normal'] & hemorrhage['Mild'] & 
                          appetite['Little'] & muscle_weakness['Normal'] & difficult_breath['Low'], 
                          dengue['Mild'])

rule23_dengue = ctrl.Rule(pain['Severe'] & temperature['High Fever'] & hemorrhage['Medium'] & 
                          appetite['None'] & muscle_weakness['Moderate'] & difficult_breath['Severe'], 
                          dengue['Strong'])

rule24_dengue = ctrl.Rule(pain['Moderate'] & temperature['Normal'] & hemorrhage['Medium'] & 
                          appetite['Little'] & muscle_weakness['Moderate'] & difficult_breath['Low'], 
                          dengue['Moderate'])

rule25_dengue = ctrl.Rule(pain['Mild'] & temperature['Low Fever'] & hemorrhage['Mild'] & 
                          appetite['Normal'] & muscle_weakness['Normal'] & difficult_breath['Low'], 
                          dengue['Mild'])

rule26_dengue = ctrl.Rule(pain['Strong'] & temperature['High Fever'] & hemorrhage['Severe'] & 
                          appetite['None'] & muscle_weakness['Severe'] & difficult_breath['Severe Low'], 
                          dengue['Strong'])

rule27_dengue = ctrl.Rule(pain['Moderate'] & temperature['High Fever'] & hemorrhage['Mild'] & 
                          appetite['Little'] & muscle_weakness['Moderate'] & difficult_breath['Moderate'], 
                          dengue['Moderate'])

rule28_dengue = ctrl.Rule(pain['Severe'] & temperature['Low Fever'] & hemorrhage['Medium'] & 
                          appetite['Little'] & muscle_weakness['Moderate'] & difficult_breath['Severe Low'], 
                          dengue['Strong'])

rule29_dengue = ctrl.Rule(pain['No pain'] & temperature['Normal'] & hemorrhage['Mild'] & 
                          appetite['Normal'] & muscle_weakness['Normal'] & difficult_breath['Low'], 
                          dengue['Mild'])

rule30_dengue = ctrl.Rule(pain['Strong'] & temperature['High Fever'] & hemorrhage['Medium'] & 
                          appetite['None'] & muscle_weakness['Severe'] & difficult_breath['Severe'], 
                          dengue['Strong'])


# Create the fuzzy control system with the modified rules
disease_ctrl = ctrl.ControlSystem([rule1_dengue, rule2_dengue, rule3_dengue, rule4_dengue, rule5_dengue, 
                                   rule6_dengue, rule7_dengue, rule8_dengue, rule9_dengue, rule10_dengue, 
                                   rule11_dengue, rule12_dengue, rule13_dengue, rule14_dengue, rule15_dengue, 
                                   rule16_dengue, rule17_dengue, rule18_dengue, rule19_dengue, rule20_dengue, 
                                   rule21_dengue, rule22_dengue, rule23_dengue, rule24_dengue, rule25_dengue, 
                                   rule26_dengue, rule27_dengue, rule28_dengue, rule29_dengue, rule30_dengue])


# Simulación del sistema difuso
@app.route('/disease_risk', methods=['POST'])
def disease_risk():
    data = request.get_json()
    # Procesar los datos y calcular los riesgos de las enfermedades
    risks = calculate_disease_risk(data)
    return jsonify({'disease_risks': risks, 'data': data})

    
def calculate_disease_risk(data):
    # Crear una simulación para el sistema de control difuso
    disease_sim = ctrl.ControlSystemSimulation(disease_ctrl)
    
    # Asignar los valores de entrada
    disease_sim.input['pain'] = data['pain']
    disease_sim.input['temperature'] = data['temperature']
    disease_sim.input['hemorrhage'] = data['hemorrhage']
    disease_sim.input['appetite'] = data['appetite']
    disease_sim.input['muscle_weakness'] = data['muscle_weakness']
    disease_sim.input['difficult_breath'] = data['difficult_breath']
    
    # Imprimir los valores de entrada
    print(f"Entradas del sistema difuso: {data}")

    # Computar las inferencias para todas las enfermedades
    disease_sim.compute()
    
    # Extraer los valores de riesgo para todas las enfermedades
    risks = {
        'dengue_risk': disease_sim.output.get('dengue', None),
    }

    # Verificar si alguna salida es nula
    print(f"Riesgos calculados: {risks}")
    
    # Devolver los riesgos calculados
    return risks

if __name__ == '__main__':
    app.run(debug=True, port=5000)
