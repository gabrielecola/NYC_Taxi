#!/bin/bash
set -e

# 1️⃣ Make sure we are in the repo root
if [ ! -d ".git" ]; then
  echo "❌ This is not a Git repository."
  exit 1
fi

# 2️⃣ Add mlruns to .gitignore (if not already)
if ! grep -q "^mlruns/" .gitignore 2>/dev/null; then
  echo "mlruns/" >> .gitignore
  git add .gitignore
  git commit -m "chore: ignore mlruns in .gitignore" || echo "ℹ️ No changes to commit for .gitignore"
fi

# 3️⃣ Install git-filter-repo if missing
if ! command -v git-filter-repo &>/dev/null; then
  echo "📦 Installing git-filter-repo..."
  pip install git-filter-repo
fi

# 4️⃣ Clean mlruns from all history
echo "🧹 Removing mlruns from Git history..."
git filter-repo --path-glob 'mlruns/**' --invert-paths --force

# 5️⃣ Push cleaned history
echo "🚀 Force-pushing cleaned history to origin/main..."
git push origin main --force

echo "✅ Done! 'mlruns/' is removed from history and ignored going forward."
