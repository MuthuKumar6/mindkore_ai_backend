function decide(observation) {
  const { stock, soldToday } = observation;

  const salesRate = soldToday / stock;

  if (salesRate < 0.05) {
    return "DECREASE_PRICE";
  }

  if (salesRate > 0.2) {
    return "INCREASE_PRICE";
  }

  return "KEEP_PRICE";
}

module.exports = decide;
