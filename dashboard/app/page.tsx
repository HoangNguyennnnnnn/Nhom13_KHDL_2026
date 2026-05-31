import { getChurnAlerts, getProductClusters } from "../lib/api";

export default async function DashboardPage() {
  const [products, churnAlerts] = await Promise.all([
    getProductClusters().catch(() => []),
    getChurnAlerts().catch(() => [])
  ]);

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <h1>TGDD Analytics Dashboard</h1>
          <p>Sentiment, product segments, churn risk, and sales forecasting.</p>
        </div>
      </header>

      <section className="metrics">
        <article>
          <span>Products</span>
          <strong>{products.length}</strong>
        </article>
        <article>
          <span>Churn alerts</span>
          <strong>{churnAlerts.length}</strong>
        </article>
        <article>
          <span>Forecast horizon</span>
          <strong>7 days</strong>
        </article>
      </section>

      <section className="grid">
        <div className="panel">
          <h2>Product Clusters</h2>
          <table>
            <thead>
              <tr>
                <th>Product</th>
                <th>Brand</th>
                <th>Cluster</th>
                <th>Price</th>
              </tr>
            </thead>
            <tbody>
              {products.slice(0, 20).map((product) => (
                <tr key={product.Product_ID}>
                  <td>{product.Product_ID}</td>
                  <td>{product.Brand ?? "-"}</td>
                  <td>{product.cluster_name ?? product.cluster_id ?? "-"}</td>
                  <td>{product.Discounted_Price?.toLocaleString("vi-VN") ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="panel">
          <h2>Churn Alerts</h2>
          <table>
            <thead>
              <tr>
                <th>User</th>
                <th>Email</th>
                <th>Segment</th>
                <th>Risk</th>
              </tr>
            </thead>
            <tbody>
              {churnAlerts.slice(0, 20).map((customer) => (
                <tr key={customer.User_ID}>
                  <td>{customer.User_ID}</td>
                  <td>{customer.email ?? "-"}</td>
                  <td>{customer.rfm_segment ?? "-"}</td>
                  <td>{Math.round(customer.churn_probability * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
