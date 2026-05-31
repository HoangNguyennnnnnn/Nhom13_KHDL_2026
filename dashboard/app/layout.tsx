import "./styles.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "TGDD Analytics Dashboard",
  description: "Customer sentiment, churn, clustering, cross-sell, and forecasting"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
