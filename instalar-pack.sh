#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(
    CDPATH= cd -- "$(dirname -- "$0")"
    pwd -P
)
ROOT=$SCRIPT_DIR
SRC=$ROOT/md
AUTO_CONFIRM=0
DEST=

if [ "${1-}" = "-y" ] || [ "${1-}" = "--yes" ]; then
    AUTO_CONFIRM=1
    shift
fi

if [ ! -f "$SRC/README.md" ]; then
    printf '[ERROR] No se encontro la carpeta del pack:\n'
    printf '        %s\n' "$SRC"
    exit 1
fi

if [ "${1-}" = "" ]; then
    DEST=$ROOT
    AUTO_CONFIRM=1
    printf '\n'
    printf 'Instalador del pack portable de instrucciones para IA\n'
    printf '\n'
    printf 'Origen:\n'
    printf '  %s\n' "$SRC"
    printf 'Destino automatico (directorio actual):\n'
    printf '  %s\n' "$DEST"
    printf '\n'
else
    DEST=$1
    shift
fi

if [ "${1-}" = "-y" ] || [ "${1-}" = "--yes" ]; then
    AUTO_CONFIRM=1
fi

if [ -z "${DEST:-}" ]; then
    printf '[ERROR] No se indico carpeta destino.\n'
    exit 1
fi

if [ -e "$DEST" ] && [ ! -d "$DEST" ]; then
    printf '[ERROR] El destino existe pero no es una carpeta:\n'
    printf '        %s\n' "$DEST"
    exit 1
fi

mkdir -p "$DEST"

DEST_ABS=$(
    CDPATH= cd -- "$DEST"
    pwd -P
)
ROOT_ABS=$ROOT
SRC_ABS=$SRC

if [ "$DEST_ABS" = "$SRC_ABS" ]; then
    printf '[ERROR] El destino no puede ser la propia carpeta md.\n'
    exit 1
fi

printf '\n'
printf 'Se copiara el contenido de:\n'
printf '  %s\n' "$SRC_ABS"
printf 'a:\n'
printf '  %s\n' "$DEST_ABS"
printf '\n'

if [ "$AUTO_CONFIRM" = "1" ]; then
    printf 'Confirmacion automatica activada.\n'
else
    printf 'Continuar? [s/N]: '
    IFS= read -r ANSWER
    case "$ANSWER" in
        s|S|si|SI|Si|y|Y|yes|YES|Yes)
            ;;
        *)
            printf 'Cancelado.\n'
            exit 0
            ;;
    esac
fi

cp -a "$SRC"/. "$DEST_ABS"/

printf '\n'
printf '[OK] Pack instalado correctamente.\n'
printf '\n'
printf 'Siguiente paso recomendado:\n'
printf '%s\n' '- Abre el proyecto destino en tu herramienta de IA.'
printf '%s\n' '- Pide: Resume las reglas locales que estas siguiendo en este repo.'
printf '%s\n' '- Si la herramienta ignora archivos locales, pega PROMPT-MANUAL.md.'
