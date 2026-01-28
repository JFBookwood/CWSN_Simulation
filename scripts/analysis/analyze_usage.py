import os

import re

scripts_dir = 'scripts'

script_files = {f[:-3]: f for f in os.listdir(scripts_dir) if f.endswith('.py') and f != '__init__.py'}

used_scripts = set()

imported_modules = {}

for script_name, script_file in script_files.items():

    path = os.path.join(scripts_dir, script_file)

    try:

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:

            content = f.read()

        imports = re.findall(r'from (\w+) import|import (\w+)', content)

        if imports:

            imported_modules[script_name] = set()

            for imp in imports:

                mod = imp[0] or imp[1]

                if mod in script_files:

                    imported_modules[script_name].add(mod)

                    used_scripts.add(mod)

    except Exception as e:

        print(f'Error reading {script_file}: {e}')

print('Used scripts:')

for s in sorted(used_scripts):

    print(f'  {s}')

print('\nPotentially unused scripts:')

for script_name in sorted(script_files.keys()):

    if script_name not in used_scripts:

        print(f'  {script_name}')
