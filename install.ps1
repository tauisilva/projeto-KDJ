# Comando para habilitar caminhos longos no Registro do Windows
New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force

# Comando para instalar as dependências a partir do arquivo requirements.txt
pip install -r requirements.txt
