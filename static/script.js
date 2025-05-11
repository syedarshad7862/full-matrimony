function confirmDelete() {
    return confirm("Are you sure you want to delete your biodata? This action cannot be undone.")
}
console.log("this is working")

function editProfile(id) {
    console.log("running inside edit fn")
    window.location.href = '/edit-profiles/' + id;
}

// GetRequest method

const GetRequest = async(url) => {
    // fetching data from url 
    let response = await fetch(url);

    // convert response into json
    let data = response.json()
    return data
}

    // async function loadProfiles() {
    //   const minCompletion = document.getElementById('filter-select').value;

    //   try {
    //     const response = await fetch(`/api/filtered-profiles?min_completion=${minCompletion}`);
    //     const data = await response.json();
    //     console.log(data);
    //     // document.getElementById('profile-container').innerHTML = html;
    //   } catch (err) {
    //     console.error("Error loading profiles:", err);
    //   }

        
    // }

function showLoader() {
    try {
        document.getElementById("loader").style.display = "block";
    } catch (err) {
        console.error("Error in showLoader:", err);
    }
}

function disableBtn(){
    try {
        let btn = document.getElementById("btn-show")
        btn.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
        Loading...`
        btn.disabled = true;
        console.log("btn is fire");
        
    } catch (error) {
        console.errer("error in disableBtn: ", error)
    }
}

function fetchFullProfile(profileId) {
  fetch(`/api/full-details?profile_id=${profileId}`)
    .then(response => {
      if (!response.ok) throw new Error("Profile not found");
      return response.json();
    })
    .then(profile => {
      document.getElementById("profileModalLabel").textContent = `${profile.full_name}'s Full Profile`;
      document.getElementById("profileModalBody").innerHTML = `
        <p><strong>Age:</strong> ${profile.age}</p>
        <p><strong>date of birth:</strong> ${profile.date_of_birth}</p>
        <p><strong>Gender:</strong> ${profile.gender}</p>
        <p><strong>Height:</strong> ${profile.height}</p>
        <p><strong>Occupation:</strong> ${profile.occupation}</p>
        <p><strong>Education:</strong> ${profile.education}</p>
        <p><strong>Marital Status:</strong> ${profile.marital_status}</p>
        <p><strong>Maslak/Sect:</strong> ${profile.maslak_sect}</p>
        <p><strong>Education:</strong> ${profile.contact_no}</p>
        <p><strong>Native Place:</strong> ${profile.native_place}</p>
        <p><strong>Family Background:</strong> ${profile.family_background}</p>
        <p><strong>Preferences:</strong> ${profile.preferences}</p>
      `;
    })
    .catch(error => {
      document.getElementById("profileModalBody").innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
    });
}

