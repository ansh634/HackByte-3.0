document.getElementById('scanBtn').addEventListener('click', async () => {
  const barcode = document.getElementById('barcode').value;
  if (!barcode) return;

  // Simulate fetching nutritional data
  const mockDatabase = {
    "1234567890123": {
      name: "Energy Bar",
      nutrients: {
        Calories: 250,
        Protein: 10,
        Carbs: 30,
        Fats: 8
      }
    },
    "9876543210987": {
      name: "Protein Shake",
      nutrients: {
        Calories: 180,
        Protein: 20,
        Carbs: 10,
        Fats: 5
      }
    }
  };

  const product = mockDatabase[barcode];
  const resultDiv = document.getElementById('result');

  if (!product) {
    resultDiv.innerHTML = "Product not found.";
    return;
  }

  let html = `<strong>${product.name}</strong><ul>`;
  for (let [nutrient, value] of Object.entries(product.nutrients)) {
    html += `<li>${nutrient}: ${value}g (Monthly: ${(value * 30).toFixed(2)}g)</li>`;
  }
  html += `</ul>`;
  resultDiv.innerHTML = html;
});