#!/usr/bin/env python3
"""
Script de prueba para validar todos los modelos optimizados
"""

import sys
import traceback

def test_imports():
    """Probar que todas las librerías se importan correctamente"""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb
        import optuna
        from optuna.samplers import TPESampler
        import psutil
        
        print("✅ Todas las librerías se importan correctamente")
        print(f"   NumPy: {np.__version__}")
        print(f"   Pandas: {pd.__version__}")
        print(f"   XGBoost: {xgb.__version__}")
        print(f"   Optuna: {optuna.__version__}")
        print(f"   CPU Cores: {psutil.cpu_count()}")
        print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        return True
    except Exception as e:
        print(f"❌ Error en imports: {e}")
        return False

def test_churn_model():
    """Probar el modelo de fuga de clientes"""
    try:
        print("\n🔍 Probando modelo de fuga de clientes...")
        with open('modelo_fuga_clientes.py', 'r', encoding='utf-8') as f:
            exec(f.read())
        print("✅ Modelo de fuga funcionando correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en modelo de fuga: {e}")
        traceback.print_exc()
        return False

def test_cobranza_model():
    """Probar el modelo de cobranza"""
    try:
        print("\n🔍 Probando modelo de cobranza...")
        with open('modelo_cobranza.py', 'r', encoding='utf-8') as f:
            exec(f.read())
        print("✅ Modelo de cobranza funcionando correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en modelo de cobranza: {e}")
        traceback.print_exc()
        return False

def test_riesgo_model():
    """Probar el modelo de score de riesgo"""
    try:
        print("\n🔍 Probando modelo de score de riesgo...")
        with open('modelo_score_riesgo_bancario.py', 'r', encoding='utf-8') as f:
            exec(f.read())
        print("✅ Modelo de score de riesgo funcionando correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en modelo de score de riesgo: {e}")
        traceback.print_exc()
        return False

def main():
    """Función principal de prueba"""
    print("🚀 Iniciando pruebas de modelos bancarios optimizados...")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Modelo Fuga", test_churn_model),
        ("Modelo Cobranza", test_cobranza_model),
        ("Modelo Riesgo", test_riesgo_model)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📊 Ejecutando: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📈 RESUMEN DE PRUEBAS:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("💡 Los notebooks optimizados están listos para usar")
    else:
        print("⚠️ Algunas pruebas fallaron")
        print("💡 Revisa los errores arriba")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
