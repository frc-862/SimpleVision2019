// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
// California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
//
// This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
// redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
// Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.  You should have received a copy of the GNU General Public License along with this program;
// if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
//
// Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
// Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>
#include <jevois/Image/RawImageOps.H>
#include <jevoisbase/Components/ObjectDetection/BlobDetector.H>
#include <jevois/Image/RawImageOps.H>
#include <opencv2/imgproc/imgproc.hpp>
#include <string.h>
#include <jevois/Debug/Timer.H>


// icon by Catalin Fertu in cinema at flaticon

//! JeVois sample module
/*! This module is provided as an example of how to create a new standalone module.

JeVois provides helper scripts and files to assist you in programming new modules, following two basic formats:

- if you wish to only create a single module that will execute a specific function, or a collection of such modules
where there is no shared code between the modules (i.e., each module does things that do not relate to the other
  modules), use the skeleton provided by this sample module. Here, all the code for the sample module is compiled
  into a single shared object (.so) file that is loaded by the JeVois engine when the corresponding video output
  format is selected by the host computer.

  - if you are planning to write a collection of modules with some shared algorithms among several of the modules, it
  is better to first create machine vision Components that implement the algorithms that are shared among several of
  your modules. You would then compile all your components into a first shared library (.so) file, and then compile
  each module into its own shared object (.so) file that depends on and automatically loads your shared library file
  when it is selected by the host computer. The jevoisbase library and collection of components and modules is an
  example for how to achieve that, where libjevoisbase.so contains code for Saliency, ObjectRecognition, etc
  components that are used in several modules, and each module's .so file contains only the code specific to that
  module.

  @author Sample Author

  @videomapping YUYV 640 480 28.5 YUYV 640 480 28.5 Lightning SimpleVision
  @email sampleemail\@samplecompany.com
  @address 123 First Street, Los Angeles, CA 90012
  @copyright Copyright (C) 2017 by Sample Author
  @mainurl http://samplecompany.com
  @supporturl http://samplecompany.com/support
  @otherurl http://samplecompany.com/about
  @license GPL v3
  @distribution Unrestricted
  @restrictions None */
  class SimpleVision : public jevois::Module
  {
    public:
    //! Default base class constructor ok
    // using jevois::Module::Module;

    SimpleVision(std::string name) : jevois::Module::Module(name) {
      itsDetector = addSubComponent<BlobDetector>("detector");
    }

    //! Virtual destructor for safe inheritance
    virtual ~SimpleVision() { }

    //! Processing function, no USB video output
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image. Any resolution and format ok:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width;

      // Convert input image to BGR24, then to HSV:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
      cv::Mat imghsv; cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Detect blobs and get their contours:
      auto contours = itsDetector->detect(imghsv);

      sendObjects(w, contours);
    }


    void sendRectangle(int count, const cv::Rect& r) {
      sendSerial("SV" + std::to_string(count) + " " + 
        std::to_string(r.x + 0.5F * r.width) + " " + 
        std::to_string(r.y + 0.5F + r.height) + " " +
        std::to_string(r.width) + " " + 
        std::to_string(r.height));
    }

    void sendObjects(int width, const std::vector<std::vector<cv::Point> >& contours) {
      cv::Rect r;

      switch (contours.size()) {
        case 0:
          sendSerial("SV0 0 0 0 0");
          break;

        case 1:
          sendRectangle(1, cv::boundingRect(contours.front()));
          break;

        case 2:
          r = cv::boundingRect(contours.front()) | cv::boundingRect(contours.back());
          sendRectangle(2, r);
          break;

        default:
          // sendSerial("SV" + std::to_string(contours.size()));
          // find center most rectangles and return them...
          {
            std::vector<cv::Rect> rects;
            rects.resize(contours.size());
            std::transform(contours.cbegin(), contours.cend(), rects.begin(), [](auto c) { return cv::boundingRect(c); });
            int middle = width / 2;
            std::sort(rects.begin(), rects.end(), [middle](auto r1, auto r2) { 
              return std::abs(middle - (r1.x + 0.5F * r1.width)) > 
                     std::abs(middle - (r2.x + 0.5F * r2.width)); 
            });
            sendRectangle(contours.size(), rects[0] | rects[1]);
          }  
          break;
      }

    }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing");

      // Wait for next available camera image. Any resolution ok, but require YUYV since we assume it for drawings:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      timer.start();

      // While we process it, start a thread to wait for output frame and paste the input image into it:
      jevois::RawImage outimg; // main thread should not use outimg until paste thread is complete
      auto paste_fut = std::async(std::launch::async, [&]() {
        outimg = outframe.get();
        outimg.require("output", w, h + 14, inimg.fmt);
        jevois::rawimage::paste(inimg, outimg, 0, 0);
        jevois::rawimage::writeText(outimg, "Lightning Deepspace Tracker", 3, 3, jevois::yuyv::White);
        jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, 0x8000);
      });

      // Convert input image to BGR24, then to HSV:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
      cv::Mat imghsv; cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);

      // Detect blobs and get their contours:
      auto contours = itsDetector->detect(imghsv);

      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();

      // Draw all detected contours in a thread:
      std::future<void> draw_fut = std::async(std::launch::async, [&]() {
        // We reinterpret the top portion of our YUYV output image as an opencv 8UC2 image:
        cv::Mat outuc2 = jevois::rawimage::cvImage(outimg); // pixel data shared
        cv::drawContours(outuc2, contours, -1, jevois::yuyv::LightPurple, 2, 8);
      });

      // Send a serial message and draw a circle for each detected blob:
      sendObjects(w, contours);
      // for (auto const & c : contours)
      // {
      //   cv::Moments moment = cv::moments(c);
      //   double const area = moment.m00;
      //   int const x = int(moment.m10 / area + 0.4999);
      //   int const y = int(moment.m01 / area + 0.4999);
      //   jevois::rawimage::drawCircle(outimg, x, y, 20, 1, jevois::yuyv::LightGreen);
      // }

      // Show number of detected objects:
      jevois::rawimage::writeText(outimg, "Detected " + std::to_string(contours.size()) + " objects.",
        3, h + 2, jevois::yuyv::White);

        // Show processing fps:
        std::string const & fpscpu = timer.stop();
        jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

        // Wait until all contours are drawn, if they had been requested:
        draw_fut.get();

        // Send the output image with our processing results to the host over USB:
        outframe.send();
    }

    private:
    std::shared_ptr<BlobDetector> itsDetector;
  };

  // Allow the module to be loaded as a shared object (.so) file:
  JEVOIS_REGISTER_MODULE(SimpleVision);
