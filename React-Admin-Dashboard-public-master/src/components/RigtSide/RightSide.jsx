import React from "react";
import CustomerReview from "../CustomerReview/CustomerReview";
// import Updates from "../Updates/Updates";
import "./RightSide.css";

const RightSide = () => {
  return (
    <div className="RightSide">
      <div>
        <h2><b>DashBoard OverView</b></h2>
        <p><b>A dashboard is a user interface that provides a centralized, visual representation of key information, metrics, or data, often presented in the form of charts, graphs, tables, and summaries. Dashboards are used in various fields like business, IT, healthcare, and analytics to monitor performance, track trends, and make data-driven decisions.</b></p>
        {/* <Updates /> */}
      </div>
      <div>
        <h3>OverAll</h3>
        <CustomerReview />
      </div>
    </div>
  );
};

export default RightSide;
