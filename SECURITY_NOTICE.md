# 🔒 SECURITY NOTICE - READ FIRST

## ⚠️ CRITICAL: PyTorch Version Requirement

**You MUST use PyTorch 2.6.0 or later**

### Why This Matters

Earlier versions of PyTorch (before 2.6.0) have a serious security vulnerability in `torch.load()` that can allow malicious code execution when loading model checkpoints. Even with `weights_only=True`, the vulnerability exists.

### Vulnerability Details

- **Affected versions:** PyTorch < 2.6.0
- **Issue:** `torch.load()` security vulnerability
- **Risk:** Remote code execution when loading untrusted models
- **Fix:** Upgrade to PyTorch 2.6.0+

### Installation

#### Option 1: Install from requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

#### Option 2: Manual Installation

**For CPU:**
```bash
pip install torch>=2.6.0 torchvision torchaudio
```

**For GPU (CUDA 11.8):**
```bash
pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For GPU (CUDA 12.1):**
```bash
pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For GPU (CUDA 12.4 - Latest):**
```bash
pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Verify Your Installation

```bash
python check_system.py
```

Or manually check:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

You should see version 2.6.0 or higher.

### If You See Version Errors

If you get an error like:
```
ValueError: serious vulnerability issue in torch.load even with weights_only=True
we now require users to upgrade torch to at least v2.6
```

**Solution:**
```bash
# Uninstall old version
pip uninstall torch torchvision torchaudio

# Install latest secure version
pip install torch>=2.6.0 torchvision torchaudio
```

### Additional Security Best Practices

1. **Only load models from trusted sources**
2. **Verify model checksums before loading**
3. **Keep PyTorch updated to the latest stable version**
4. **Never load models from untrusted URLs or unknown sources**
5. **Run training in isolated environments when possible**

### What This Project Does

This TTS training project:
- ✅ Requires PyTorch 2.6.0+ in requirements.txt
- ✅ Checks PyTorch version at runtime (check_system.py)
- ✅ Uses secure model loading practices
- ✅ Only loads from official Hugging Face repositories
- ✅ Saves checkpoints with proper permissions

### Questions?

If you have questions about this security requirement:
1. Check the official PyTorch security advisory
2. Review the installation guide
3. Run `python check_system.py` to verify your setup

### References

- PyTorch Official Documentation: https://pytorch.org/get-started/locally/
- Hugging Face Transformers Security: https://huggingface.co/docs/transformers/security

---

**Do not proceed with training until you have PyTorch 2.6.0 or later installed.**

This is not optional - it's a critical security requirement.
