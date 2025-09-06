import { SolendMarket } from "@solendprotocol/solend-sdk";
import { Connection, PublicKey } from "@solana/web3.js";

const connection = new Connection("https://api.mainnet-beta.solana.com");
const market = await SolendMarket.initialize(
  connection,
  "production",     // or 'devnet'
  new PublicKey("<marketAddress>") // optional
);

// Read reserves & parameters
await market.loadReserves();
const usdcReserve = market.reserves.find(r => r.config.liquidityToken.symbol === "USDC");

// Example: propose a param change (pseudo; actual admin ops require authority + tx)
console.log(usdcReserve.config.optimalUtilizationRate);
