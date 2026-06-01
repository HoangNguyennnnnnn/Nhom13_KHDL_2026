import { getProductClusters } from "../lib/api";
import DashboardClient from "./DashboardClient";
import fs from "fs";
import path from "path";

export default async function DashboardPage() {
  const products = await getProductClusters().catch(() => []);
  
  // Read static JSON files from the workspace directory
  let aspects = {};
  let elasticity = {};
  let teencode = {};

  try {
    const aspectsPath = path.join(process.cwd(), "../data-project/processed/aspect_sentiment.json");
    if (fs.existsSync(aspectsPath)) {
      aspects = JSON.parse(fs.readFileSync(aspectsPath, "utf-8"));
    }
  } catch (err) {
    console.error("Error reading aspect_sentiment:", err);
  }

  try {
    const elasticityPath = path.join(process.cwd(), "../data-project/processed/price_elasticity.json");
    if (fs.existsSync(elasticityPath)) {
      elasticity = JSON.parse(fs.readFileSync(elasticityPath, "utf-8"));
    }
  } catch (err) {
    console.error("Error reading price_elasticity:", err);
  }

  try {
    const teencodePath = path.join(process.cwd(), "../data-project/teencode_dict.json");
    if (fs.existsSync(teencodePath)) {
      teencode = JSON.parse(fs.readFileSync(teencodePath, "utf-8"));
    }
  } catch (err) {
    console.error("Error reading teencode_dict:", err);
  }

  return (
    <DashboardClient
      products={products}
      aspects={aspects}
      elasticity={elasticity}
      teencode={teencode}
    />
  );
}
