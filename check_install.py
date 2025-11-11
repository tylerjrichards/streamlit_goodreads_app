"""
Smoke test to verify core packages are importable and print versions.
Run:
    python check_install.py
"""
import importlib
import sys

packages = [
    "streamlit",
    "pandas",
    "numpy",
    "pyarrow",
    "openai",
    "supabase",
    "plotly",
    "python_dotenv",
]

# mapping for import names that differ from package names
import_map = {
    'python_dotenv': 'dotenv',
}

results = {}
for pkg in packages:
    import_name = import_map.get(pkg, pkg)
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, '__version__', None)
        results[pkg] = f'OK (version={ver})'
    except Exception as e:
        results[pkg] = f'MISSING or ERROR: {e}'

print('\nInstallation check results:')
for k, v in results.items():
    print(f'- {k}: {v}')

# Exit with non-zero if critical libs are missing
critical = ['streamlit', 'pandas', 'numpy']
missing_crit = [p for p in critical if not results.get(p, '').startswith('OK')]
if missing_crit:
    print('\nOne or more critical packages are missing; please install requirements.')
    sys.exit(2)
else:
    print('\nCore packages are installed.')
    sys.exit(0)
