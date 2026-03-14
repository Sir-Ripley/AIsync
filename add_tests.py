import json

with open('BigTest.ipynb', 'r') as f:
    nb = json.load(f)

test_markdown_cell = {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "## UNIT TESTS\n"
  ]
}

test_code_cell = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": [
    "import numpy as np\n",
    "import math\n",
    "\n",
    "def test_v_baryonic():\n",
    "    print(\"Running test_v_baryonic...\")\n",
    "    r = np.array([0.0, 1e-5, 1.0, 5.0, 10.0])\n",
    "    M_disk = 1e10\n",
    "    r_scale = 2.0\n",
    "    \n",
    "    # Calculate velocities\n",
    "    v = v_baryonic(r, M_disk, r_scale)\n",
    "    \n",
    "    # Test output shapes\n",
    "    assert v.shape == r.shape, f\"Expected shape {r.shape}, got {v.shape}\"\n",
    "    \n",
    "    # Test values are valid (no NaNs, non-negative)\n",
    "    assert np.all(v >= 0), \"Velocities must be non-negative\"\n",
    "    assert not np.any(np.isnan(v)), \"Velocities must not contain NaNs\"\n",
    "    \n",
    "    # Test edge case: zero mass disk should result in zero velocity\n",
    "    v_zero_mass = v_baryonic(r, 0.0, r_scale)\n",
    "    assert np.all(v_zero_mass == 0.0), \"Zero mass should yield zero velocity\"\n",
    "    \n",
    "    # Return calculated result for notebook testing, avoiding just printing\n",
    "    return v\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    res = test_v_baryonic()\n",
    "    print(\"v_baryonic tests passed! Output:\", res)\n"
  ]
}

nb['cells'].append(test_markdown_cell)
nb['cells'].append(test_code_cell)

with open('BigTest.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
