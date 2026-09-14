"""Convierte uno o varios CSV grandes a Parquet sin cargarlos enteros en RAM.

Lee cada CSV en lotes (streaming) con pyarrow y los va escribiendo
incrementalmente al Parquet de salida correspondiente.

Requisitos:
    pip install pyarrow

Uso (un archivo):
    python3 csv_to_parquet.py entrada.csv salida.parquet

Uso (todos los CSV de una carpeta -> mismo nombre .parquet al lado de cada uno):
    python3 csv_to_parquet.py /ruta/a/la/carpeta

Uso (todos los CSV de una carpeta -> otra carpeta de salida):
    python3 csv_to_parquet.py /ruta/entrada /ruta/salida
"""

import argparse
import csv
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq


def _read_header(input_path, delimiter):
    with open(input_path, newline="") as f:
        return next(csv.reader(f, delimiter=delimiter))


def convert_one(input_path, output_path, batch_size, compression, delimiter, all_strings):
    read_options = pa_csv.ReadOptions(block_size=batch_size * 200)  # bytes aprox por bloque
    parse_options = pa_csv.ParseOptions(delimiter=delimiter)

    convert_options = None
    if all_strings:
        # Fuerza todas las columnas a string: evita que el schema quede
        # fijado por el primer lote (p.ej. una columna vacía en las primeras
        # filas se infiere como null y luego revienta al aparecer texto real).
        header = _read_header(input_path, delimiter)
        convert_options = pa_csv.ConvertOptions(
            column_types={name: pa.string() for name in header}
        )

    reader = pa_csv.open_csv(
        input_path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )

    writer = None
    total_rows = 0
    try:
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break

            if writer is None:
                writer = pq.ParquetWriter(output_path, batch.schema, compression=compression)

            writer.write_table(pa.Table.from_batches([batch]))
            total_rows += batch.num_rows
            print(f"\r  {total_rows:,} filas procesadas...", end="", flush=True)
    finally:
        if writer is not None:
            writer.close()

    print(f"\r  Listo: {output_path.name} ({total_rows:,} filas)" + " " * 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", help="CSV de entrada, o carpeta con varios CSV")
    parser.add_argument("output_path", nargs="?", default=None, help="Parquet de salida, o carpeta de salida si input_path es una carpeta")
    parser.add_argument("--batch-size", type=int, default=200_000, help="Filas aproximadas por lote (default: 200000)")
    parser.add_argument("--compression", default="snappy", choices=["snappy", "gzip", "zstd", "brotli", "none"], help="Compresión del Parquet (default: snappy)")
    parser.add_argument("--delimiter", default=",", help="Delimitador del CSV (default: ',')")
    parser.add_argument("--skip-existing", action="store_true", help="Si el .parquet de salida ya existe, no reconvertir")
    parser.add_argument(
        "--infer-types",
        action="store_true",
        help="Dejar que pyarrow infiera los tipos por lote (puede fallar con columnas de tipo mixto entre lotes). Por default todas las columnas se leen como string.",
    )
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    all_strings = not args.infer_types
    input_path = Path(args.input_path)

    if input_path.is_dir():
        out_dir = Path(args.output_path) if args.output_path else input_path
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            print(f"No se encontraron .csv en {input_path}", file=sys.stderr)
            sys.exit(1)

        for i, csv_file in enumerate(csv_files, 1):
            out_file = out_dir / (csv_file.stem + ".parquet")
            print(f"[{i}/{len(csv_files)}] {csv_file.name} -> {out_file.name}")
            if args.skip_existing and out_file.exists():
                print("  Ya existe, se omite.")
                continue
            try:
                convert_one(csv_file, out_file, args.batch_size, compression, args.delimiter, all_strings)
            except Exception as e:
                print(f"  ERROR convirtiendo {csv_file.name}: {e}", file=sys.stderr)
    else:
        if not args.output_path:
            print("Falta la ruta de salida del .parquet", file=sys.stderr)
            sys.exit(1)
        try:
            convert_one(input_path, Path(args.output_path), args.batch_size, compression, args.delimiter, all_strings)
        except FileNotFoundError:
            print(f"Error: no se encontró el archivo {input_path}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
