import pyiqa

print("=" * 60)
print("PYIQA VERSION")
print("=" * 60)
print(pyiqa.__version__)

print("\n" + "=" * 60)
print("AVAILABLE IQA MODELS")
print("=" * 60)

models = pyiqa.list_models()

for model in models:
    print(model)

print("\n" + "=" * 60)

if "brisque" in models:
    print("✓ BRISQUE is available")
else:
    print("✗ BRISQUE NOT found")

if "niqe" in models:
    print("✓ NIQE is available")
else:
    print("✗ NIQE NOT found")

print("=" * 60)

try:
    brisque = pyiqa.create_metric(
        "brisque",
        device="cpu"
    )
    print("✓ BRISQUE loaded successfully")
except Exception as e:
    print("✗ BRISQUE load failed")
    print(e)

print()

try:
    niqe = pyiqa.create_metric(
        "niqe",
        device="cpu"
    )
    print("✓ NIQE loaded successfully")
except Exception as e:
    print("✗ NIQE load failed")
    print(e)

print("=" * 60)