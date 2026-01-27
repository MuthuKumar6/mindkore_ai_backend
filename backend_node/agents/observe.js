function observe(product) {
  return {
    price: product.price,
    stock: product.stock,
    soldToday: product.soldToday
  };
}

module.exports = observe;
