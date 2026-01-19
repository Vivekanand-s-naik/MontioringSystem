// Sidebar imports
//Get The data from cloud like name ,marks attedance
import {
  UilEstate,
  UilClipboardAlt,
  UilUsersAlt,
  UilPackage,
  UilChart,
  UilSignOutAlt,
} from "@iconscout/react-unicons";

// Analytics Cards imports
import { UilUsdSquare, UilMoneyWithdrawal } from "@iconscout/react-unicons";
import { keyboard } from "@testing-library/user-event/dist/keyboard";

// Recent Card Imports
// import img1 from "../imgs/img1.png";
// import img2 from "../imgs/img2.png";
// import img3 from "../imgs/img3.png";

// Sidebar Data
export const SidebarData = [
  {
    icon: UilEstate,
    heading: "Dashboard",
  },
  
  {
    icon: UilChart,
    heading: 'Analytics'
  },
];
// Analytics Cards Data
// export const cardsData = [
//   {
//     title: "Marks",
//     color: {
//       backGround: "linear-gradient(180deg, #bb67ff 0%, #c484f3 100%)",
//       boxShadow: "0px 20px 30px 0px #e0c6f5",
//     },
//     barValue: 70,
//     value: "",
//     png: UilUsdSquare,
//     series: [
//       {
//         name: "",//get Name from Cloud 
//         data: [31, 40, 28, 51, 42, 109, 100],//Data To Visualise 
//       },
//     ],
//   },
//   {
//     title: "Attendence",
//     color: {
//       backGround: "linear-gradient(180deg,rgb(145, 215, 255) 0%,rgb(183, 146, 252) 100%)",
//       boxShadow: "0px 20px 30px 0px #FDC0C7",
//     },
//     barValue: 80,
//     value: "",
//     png: UilMoneyWithdrawal,
//     series: [
//       {
//         name: "",
//         data: [10, 100, 50, 70, 80, 30, 40],
//       },
//     ],
//   },
//   {
//     title: "Attendence shortage",
//     color: {
//       backGround:
//         "linear-gradient(rgb(248, 212, 154) -146.42%, rgb(255 202 113) -46.42%)",
//       boxShadow: "0px 20px 30px 0px #F9D59B",
//     },
//     barValue: 60,
//     value: "4,270",
//     png: UilClipboardAlt,
//     series: [
//       {
//         name: "",
//         data: [10, 25, 15, 30, 12, 15, 20],
//       },
//     ],
//   },
// ];

export const cardsData = [
  {
    title: "Marks",
    color: {
      backGround: "linear-gradient(180deg, #bb67ff 0%, #c484f3 100%)",
      boxShadow: "0px 10px 20px 0px #e0c6f5",
    },
    barValue: 70, // Average marks percentage
    value: "Good",
    png: UilUsdSquare,
    series: [
      {
        name: "Marks",
        data: [31, 40, 28, 51, 42, 109, 100], // Marks trend over time
      },
    ],
  },
  {
    title: "Attendance",
    color: {
      backGround: "linear-gradient(180deg,rgb(145, 215, 255) 0%,rgb(183, 146, 252) 100%)",
      boxShadow: "0px 10px 20px 0px #FDC0C7",
    },
    barValue: 80, // Attendance percentage
    value: "Consistent",
    png: UilMoneyWithdrawal,
    series: [
      {
        name: "Attendance",
        data: [10, 100, 50, 70, 80, 30, 40], // Attendance trend over time
      },
    ],
  },
  {
    title: "Engagement Score",
    color: {
      backGround:
        "linear-gradient(180deg, #f7971e 0%, #ffd200 100%)",
      boxShadow: "0px 10px 20px 0px #f9e3a9",
    },
    barValue: 75, // Example: Dynamic percentage of Engagement
    value: "Active",
    png: UilClipboardAlt,
    series: [
      {
        name: "Engagement",
        data: (() => {
          // Dynamically calculate engagement from Marks and Attendance
          const marksData = [31, 40, 28, 51, 42, 109, 100];
          const attendanceData = [10, 100, 50, 70, 80, 30, 40];
          return marksData.map((marks, index) => marks + (attendanceData[index] || 0));
        })(), // Sum of Marks and Attendance data
      },
    ],
  },
];



// Recent Update Card Data
// export const UpdatesData = [
//   {
//     img: img1,
//     name: "Vivekanand",
//     noti: "has score good marksd in all subjects.",
    
//   },
//   {
//     img: img2,
//     name: "siddu",
//     noti: "has score average marksd in all subjects.",
    
//   },
//   {
//     img: img3,
//     name: "vishal",
//     noti: "has score good marksd in all subjects.",

//   },
// ];
