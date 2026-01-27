const product = require("./data/product");
const observe = require("./agents/observe");
const decide = require("./agents/decide");
const act = require("./agents/act");

function runAgent() {
  const observation = observe(product);
  const decision = decide(observation);
  const updatedProduct = act(product, decision);

  console.log("Decision:", decision);
  console.log("Product State:", updatedProduct);
  console.log("--------------------------");
}

// Run every 3 seconds
setInterval(runAgent, 3000);
