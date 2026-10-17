"""Configuration commune des tests."""

import matplotlib

# Backend sans fenêtre : les tests de figures écrivent des fichiers, n'affichent rien.
matplotlib.use("Agg")
