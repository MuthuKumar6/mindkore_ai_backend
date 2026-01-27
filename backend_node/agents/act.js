function act(product, decision) {
  if (decision === "DECREASE_PRICE") {
    product.price -= 50;
  }

  if (decision === "INCREASE_PRICE") {
    product.price += 50;
  }

  return product;
}

module.exports = act;
