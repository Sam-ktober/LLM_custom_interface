import sys, os, subprocess
from pathlib import Path

# Config Windows standard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except: pass
os.environ["TORCH_COMPILE_DISABLE"] = "1"

def main():
    if len(sys.argv) < 3: sys.exit(1)
    output_path = Path(sys.argv[1])
    text = sys.argv[2]
    root = Path(os.getcwd())
    
    # Creation du dossier de sortie
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("--- FISH SPEECH 1.4.2 STABLE ---")

    # Chemins
    ckpt_dir = root / "checkpoints" / "fish-speech-1.4"
    vqgan_file = ckpt_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    
    script_gen = root / "tools" / "llama" / "generate.py"
    script_vq = root / "tools" / "vqgan" / "inference.py"

    # Nettoyage des anciens fichiers temporaires
    for f in root.glob("*.npy"):
        try: os.remove(f)
        except: pass

    # --- STEP 1: SEMANTIC ---
    print("Step 1/2: Semantic (GPU)...")
    # On utilise uniquement les options validees par ton diagnostic
    cmd_gen = [
        sys.executable, str(script_gen),
        "--text", text,
        "--checkpoint-path", str(ckpt_dir),
        "--num-samples", "1",
        "--device", "cuda"
    ]
    
    # On lance dans 'root' pour que les .npy apparaissent a la racine
    subprocess.run(cmd_gen, cwd=str(root), check=True)

    # --- STEP 2: ACOUSTIC ---
    print("Step 2/2: Acoustic (GPU)...")
    # On cherche codes_0.npy ou sample_0.npy
    codes_files = list(root.glob("codes_*.npy")) + list(root.glob("sample_*.npy"))
    
    if not codes_files:
        print("Error: No codes found after Step 1.")
        sys.exit(1)
        
    latest_codes = sorted(codes_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"Using codes: {latest_codes.name}")
    
    cmd_vq = [
        sys.executable, str(script_vq),
        "-i", str(latest_codes),
        "--checkpoint-path", str(vqgan_file),
        "--output-path", str(output_path),
        "--device", "cuda"
    ]
    subprocess.run(cmd_vq, check=True)

    if output_path.exists():
        print(f"SUCCESS: {output_path.name}")

if __name__ == "__main__":
    main()